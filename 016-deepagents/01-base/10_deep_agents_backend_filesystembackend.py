# todo: 将重要信息主动存储到backend -> file -> 文件中! 跨线程提问 -> 数据
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import os
from deepagents import create_deep_agent
from deepagents.backends import StateBackend, StoreBackend, FilesystemBackend, CompositeBackend
from pathlib import Path

# 1. 加载配置文件
load_dotenv(override=True)

# 2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

root_dir_path = Path(__file__).resolve().parent / "agent_workspace"
root_dir_path.mkdir(parents=True, exist_ok=True)

file_backend = FilesystemBackend( # FilesystemBackend 真实磁盘目录
    root_dir=root_dir_path,
    virtual_mode=True)  # 虚拟路径模式: 文件操作被限制在root_dir内, 阻止`..`/`~`穿越和绝对路径逃逸, 但不做沙箱隔离

# 检查点
checkpointer = InMemorySaver()

"""
自己总结：  thread_id 隔离会话状态，但 FilesystemBackend 的文件是全局共享的，任何线程都能读写。
"""

# 3. deepagent
deep_agent = create_deep_agent(
    model=model,
    tools=[],
    subagents=[],
    system_prompt="""
      你是一个综合助手! 用户输入个人信息,都要使用backend主动存储!
      存储的文件名: info.txt中!!
    """,
    checkpointer=checkpointer,
    backend=file_backend
)


config_1 = {"configurable": {"thread_id": "007"}}
result = deep_agent.invoke(input={
    "messages": [
        HumanMessage(content="我的名字叫大风哥,年龄是23岁,我喜欢38号!")
    ]
},config=config_1)
print(f"第一次执行结果:{result['messages'][-1].content}")


config_2 = {"configurable": {"thread_id": "008"}}
result_2 = deep_agent.invoke(input={
    "messages": [
        HumanMessage(content="我是谁?我今年多大?我喜欢啥?")
    ]
}, config=config_2)
print(f"第二次执行结果:{result_2['messages'][-1].content}")

print(checkpointer)
print(vars(checkpointer))
