"""
DeepAgents stream 模式下的 HITL（人工审批）示例
核心功能：
1. 高危工具调用前触发人工审批
2. 使用 stream() 流式获取执行过程
3. 使用 Command(resume=...) 恢复执行
"""

import os
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 1. 加载配置文件
load_dotenv(override=True)

#2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# ======================== 1. 定义工具 ========================

@tool
def delete_database(table_name: str):
    """高危操作：删除数据库表"""
    print(f"[工具执行] 删除数据库表: {table_name}")
    return f"已成功删除数据库表：{table_name}"


@tool
def delete_file(file_name: str):
    """高危操作：删除文件"""
    print(f"[工具执行] 删除文件: {file_name}")
    return f"已成功删除文件：{file_name}"


@tool
def select_data(table_name: str):
    """普通操作：查询表数据（无需审批）"""
    print(f"[工具执行] 查询表数据: {table_name}")
    return f"查询成功，表名：{table_name}"


# ======================== 2. 初始化 Agent ========================

checkpointer = InMemorySaver()
config = {"configurable":{"thread_id":"007"}}


# 创建多智能配置人机交互拦截
main_agent = create_deep_agent(
    model=model,
    system_prompt="你是一个数据库与文件管理助手，所有回答都使用中文。",
    subagents=[],
    tools=[delete_database, delete_file, select_data],
    checkpointer=checkpointer,
    interrupt_on={
        "delete_database":True,
        "delete_file":True,
        "select_data":False
    }
)




# 第一次执行
stream = main_agent.stream(
    input={
        "messages":[
            HumanMessage(content="删除user的表数据!查询product对应的表数据,删除info.txt文件!!")
        ]
    },
    config=config  # 配了 checkpointer 就必须传,否则报 thread_id 缺失错误
)

interrupts = []
for chunk in stream:
    if "__interrupt__" in chunk:
        # {"__interrupt__":[Interxxx(value={action_requests:[{name:"",args:{}},{}]})]}
        interrupts = chunk["__interrupt__"]
        print(f"本次执行触发了拦截处理!后续不在执行!")
        break
    print(chunk) # 模拟处理结果 没有被拦截


if interrupts:
    # 进行拦截的处理
    # 识别拦截动作 配置放行的列表
    action_requests = interrupts[0].value["action_requests"]
    print(action_requests)
    decisions = [{"type":"reject"},{"type":"approve"}]
    # 第二次执行
    for chunk in main_agent.stream(
            Command(resume={"decisions":decisions}),
            config=config
    ):
        print(chunk)