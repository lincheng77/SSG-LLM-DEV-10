# 将重要信息主动存储到backend -> file -> 文件中! 跨线程提问 -> 数据
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from tavily import TavilyClient
from deepagents import create_deep_agent
from deepagents.backends import StateBackend,StoreBackend,FilesystemBackend,CompositeBackend
from pathlib import Path
from langgraph.store.memory import InMemoryStore



# 1. 加载配置文件
load_dotenv(override=True)

# 2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)



store = InMemoryStore()
store_backend = StoreBackend(namespace=lambda _: ("filesystem",))
def namespace_factory(xxx):
    return ("filesystem", )
store_backend1 = StoreBackend(namespace=namespace_factory)  # 和上面store_backend 创建形式等价


# 3. deepagent
deep_agent = create_deep_agent(
    model = model,
    tools=[],
    subagents=[],
    system_prompt="""
      你是一个综合助手! 用户输入个人信息,都要使用backend主动存储!
      存储的文件名: info.txt中!!
    """,
    store=store,
    backend=store_backend # 将数据存储到   store=store,
)


# 第一次执行
config_1 = {"configurable":{"thread_id":"007"}}
result = deep_agent.invoke(input={
    "messages":[
        HumanMessage(content="我的名字叫大风哥,年龄是23岁,我喜欢38号!")
    ]
},config=config_1)
print(f"第一次执行结果:{result['messages'][-1].content}")

# 我再执行一次! 读取我的个人重要信息! [另一个线程id]


config_2 = {"configurable":{"thread_id":"008"}}
result_2 = deep_agent.invoke(input={
    "messages":[
        HumanMessage(content="我是谁?我今年多大?我喜欢啥?")
    ]
},config=config_2)
print(f"第二次执行结果:{result_2['messages'][-1].content}")


# 确定有没有存进去! store
store.put(namespace=("qita","haha"),key="name",value="erdaye")
print(store.get(namespace=("qita","haha"),key="name"))

items = store.search(("filesystem",))
for item in items:
    print(item)
    print()
    print(item.key)
    print()
    print(item.value)