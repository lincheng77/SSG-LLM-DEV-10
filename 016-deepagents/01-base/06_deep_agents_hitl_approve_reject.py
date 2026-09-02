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
# 执行添加thread_id
"""
1. interrupt_on 的暂停/恢复完全依赖 checkpointer
2. thread_id 是存档的"身份证"
"""
config = {"configurable":{"thread_id":"9527"}}

# 创建多智能体
main_agent = create_deep_agent(
    model=model,
    system_prompt="""
       一个综合智能体,拥有: delete_database 删除指定表名工具 , select_data查询表数据工具,delete_file删除指定文件工具!
       如果有对应的需求,必须调用工具执行!
    """,
    tools=[delete_database, select_data, delete_file],
    interrupt_on={
        "delete_database":True, #需要审批
        "delete_file":True,     #需要审批
        "select_data":False     #不需要审批（默认值）
    },
    # 配置一个短期记忆对象
    checkpointer=checkpointer
)



result = main_agent.invoke(input={
    "messages": [
        HumanMessage(content="先删除teacher的表数据!再查询product对应的表数据,最后删除info.txt文件!!")
    ]
},config=config)


#todo: 注意: 一旦触发了人机交互拦截的动作! 是所有的工具都不执行!! 包括查询数据表!! -> 配置放行或者拦截动作 -> 会再次规划调用流程 -> 触发
interrupts = result.get("__interrupt__")
print(interrupts)
interrupt_tool_list = interrupts[0].value['action_requests']


decisions = []
for tool in interrupt_tool_list:
    #tool =  {'name': 'delete_database', 'args': {'table_name': 'user'},
    tool_name = tool['name']
    if tool_name == 'delete_file':
        file_name = tool['args']['file_name']
        if file_name == "info.txt":
            # 拦截
            decisions.append({
                "type":"reject"
            })
        else:
            # 放行
            decisions.append({
                "type":"approve"
            })
    elif tool_name == 'delete_database':
        table_name = tool['args']['table_name']
        if table_name == "user":
            decisions.append({
                "type":"reject"
            })
        else:
            decisions.append({
                "type":"approve"
            })
    else:
        decisions.append({
            "type": "approve"
        })
# 第二次执行
result = main_agent.invoke(
    Command(resume={"decisions":decisions}),config=config)

print(result['messages'][-1].content)

