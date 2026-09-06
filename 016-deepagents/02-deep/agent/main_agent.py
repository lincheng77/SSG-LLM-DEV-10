
import shutil
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage

from api.context import set_session_dir_context, set_session_id_context, reset_session_context
from api.monitor import monitor
# tools
from tools.pdf_tools import convert_md_to_pdf
from tools.markdown_tools import generate_markdown
from tools.upload_file_read_tool import read_file_content

# subagents
from agent.subagents.ragflow_sub_agent import ragflow_sub_agent
from agent.subagents.mysql_sub_agent import mysql_sub_agent
from agent.subagents.tavily_sub_agent import tavily_sub_agent

# 系统提示词
from agent.config_load import config_dict
# 模型
from agent.llm import model
from langgraph.checkpoint.memory import InMemorySaver


project_root = Path(__file__).parent.parent.resolve()

main_agent = create_deep_agent(
    system_prompt=config_dict['main_agent']['system_prompt'],
    tools=[
        convert_md_to_pdf,
        generate_markdown,
        read_file_content
    ],
    subagents=[
        ragflow_sub_agent,
        mysql_sub_agent,
        tavily_sub_agent
    ],
    model=model,
    checkpointer=InMemorySaver()
)




def _prepare_session_environment(session_id: str):
    """
        获得用户目录（绝对目录、相对目录）、用户上传文件信息
    """

    # 1. [创建] 定义并创建会话的绝对输出路径
    session_dir = project_root / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 2. [标准化] 路径转为 POSIX 风格 (防止大模型因反斜杠产生幻觉)
    session_dir_str = str(session_dir).replace("\\", "/")

    # 3. [相对化] 获取相对路径 (用于提示词展示，如 "output/session_123")
    session_relative_dir = str(session_dir.relative_to(project_root)).replace("\\", "/")

    # 4. [上传] 检查并处理上传文件
    upload_dir = project_root / "updated" / f"session_{session_id}"
    uploaded_info = ""

    if upload_dir.exists():
        # ["名字","名字"]
        filenames = [f.name for f in upload_dir.iterdir() if f.is_file()]

        if filenames:
            for filename in filenames:
                # 核心动作：将文件从临时上传区复制到正式工作区
                shutil.copy2(upload_dir / filename, session_dir / filename)

            # 5. [构造] 生成文件列表提示词
            uploaded_info = (f"\n    [已上传文件] 已加载到工作目录:\n" +
                             "\n".join([f"    - {filename}" for filename in filenames]) +
                             "\n    请优先使用工具读取并参考这些文件。")
    return session_dir_str, session_relative_dir, uploaded_info



def _process_stream_chunk(chunk):
    for node_name, state in chunk.items():
        if not state or "messages" not in state: continue

        messages = state["messages"]
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                # Case 1: Agent 决定调用工具 (Tool Call)
                if last_msg.tool_calls:
                    for tool in last_msg.tool_calls:
                        if tool['name'] == 'task':
                            # deepagents 通过 `task` 工具派发子智能体: 上报子智能体类型与任务描述
                            monitor.report_assistant(
                                tool['args'].get('subagent_type', 'Agent'),
                                {"desc": tool['args'].get('description')}
                            )
                # Case 2: Agent 生成最终回复 (Final Answer)
                elif last_msg.content:
                    monitor.report_task_result(last_msg.content)

# ====================== 核心执行逻辑 ======================
async def run_deep_agent(task_query: str, session_id: str = None):
    """
    DeepAgents 核心执行入口 (Agent Execution Runtime)。
    """

    # 1. [ID 初始化] 确保有唯一的会话 ID
    if not session_id: session_id = str(uuid.uuid4())
    print(f"--- Start Task: {task_query} (Session: {session_id}) ---")

    # 2. [环境准备] 创建目录、处理上传文件
    session_dir_str, session_relative_dir, uploaded_info = _prepare_session_environment(session_id)

    # 3. [上下文绑定] 初始化 ContextVars (关键：隔离并发请求)
    session_id_context_token = set_session_id_context(session_id)
    session_dir_context_token = set_session_dir_context(session_dir_str)


    # 给前端推送文件夹，方便后续查询当前会话对应文件夹下的所有文件
    monitor.report_session_dir(session_dir_str)

    # 4. [运行时配置] LangChain Config (注入记忆 key)
    config = {"configurable": {"thread_id": session_id},}  # LangGraph 检查点要求此键名

    # 5. [提示词构建] 动态注入环境约束
    path_instruction = f"""
    【工作环境指令】
    工作目录: {session_relative_dir}  
    {uploaded_info}  

    规则：
    1. 新生成文件必须保存到工作目录：'{session_relative_dir}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """

    # 6. [流式执行] 启动 Agent 循环
    try:
        # astream: 异步生成器，像流水线一样逐个吐出 Agent 的思考片段
        async for chunk in main_agent.astream(
                {"messages": [{"role": "user", "content": task_query + path_instruction}]},
                config=config):
            # 实时处理每一个片段 (上报前端)
            _process_stream_chunk(chunk)
        return "Done"
    except Exception as e:
        print(f"Error: {e}")
        #TODO
        monitor._emit("error", f"Execution failed: {e}")
        return f"Error: {e}"

    finally:
        # 8. [资源清理] 必须重置 ContextVars，防止线程池复用导致的上下文污染
        if 'session_dir_context_token' in locals():
            reset_session_context(session_dir_context_token, session_id_context_token)
