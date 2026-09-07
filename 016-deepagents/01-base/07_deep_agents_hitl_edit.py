"""
DeepAgents 中断审批机制示例
核心功能：演示高危工具调用前的人工审批流程，支持删除数据库表/文件的审批控制
"""
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command  # 恢复执行的指令类型
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import InMemorySaver


# 1. 加载配置文件
load_dotenv(override=True)

#2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# ======================== 1. 定义工具函数 ========================
@tool
def delete_database(table_name: str):
    """
    高危操作：删除数据库表
    :param table_name: 要删除的表名
    :return: 操作结果提示
    """
    print(f"[工具执行] 删除表: {table_name}")
    return f"已成功删除表: {table_name}"


@tool
def select_data(table_name: str):
    """
    普通操作：查询指定表名的数据（无需审批）
    :param table_name: 要查询的表名
    :return: 操作结果提示
    """
    print(f"[工具执行] 查询指定表名数据: {table_name}")
    return f"查询数据成功：{table_name}"


@tool
def delete_file(file_name: str):
    """
    高危操作：删除文件
    :param file_name: 要删除的文件路径/名称
    :return: 操作结果提示
    """
    print(f"[工具执行] 删除文件: {file_name}")
    return f"已成功删除文件: {file_name}"




checkpointer = InMemorySaver()
config = {"configurable":{"thread_id":"9527"}}

# 创建多智能体
main_agent = create_deep_agent(
    model=model,
    system_prompt="""
       一个综合智能体,拥有: delete_database 删除指定表名工具 , select_data查询表数据工具,delete_file删除指定文件工具!
       如果有对应的需求,必须调用工具执行!
       注意:高危工具调用会经过人工审核,审核人可能修改参数、拒绝或放行。
       工具返回的结果就是最终事实,直接如实汇报即可;
       不要把被修改的参数当成自己的失误,也不要自行重试被审核变更或拒绝的操作。
       最终执行结果必须汇报！
    """,
    tools=[delete_database, select_data, delete_file],
    interrupt_on={
        "delete_database":True,
        "delete_file":True,
        "select_data":False
    },
    checkpointer=checkpointer
)

result = main_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="删除user的表数据!查询product对应的表数据,删除info.txt文件!!")
        ]
    },
    config=config
)

# 只要有拦截触发,所有工具都不会执行!!
tool_list = result.get("__interrupt__")[0].value['action_requests']
decisions = []

for tool_info in tool_list:
    # {name:"工具名",args:{table_name/file_name}}
    tool_name = tool_info['name']
    if tool_name == 'delete_database':
        table_name = tool_info['args']['table_name']
        if table_name == "user":
            # 拦截
            decisions.append(
                {
                    "type":"reject"
                }
            )
        else:
            # 放行
            decisions.append(
                {
                    "type":"approve"
                }
            )
    elif tool_name == 'delete_file':
        file_name = tool_info['args']['file_name']
        if file_name == "info.txt":
            # 编辑
            decisions.append({
                "type": "edit",
                "edited_action":{
                    "name":tool_name, # 工具名,
                    "args":{"file_name":"erdaye.txt"}
                }
            })
        else:
            # 拦截
            decisions.append({
                "type":"reject"
            })

# 二次执行
result = main_agent.invoke(
    Command(resume={"decisions":decisions}),config=config
)

print(f"最终执行结果:{result['messages'][-1].content}")