# 将重要信息主动存储到backend -> file -> 文件中! 跨线程提问 -> 数据
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os
from deepagents import create_deep_agent
from deepagents.backends import StateBackend,StoreBackend,FilesystemBackend,CompositeBackend
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


def create_name_space(runtime):
    account = runtime.context.account
    return ("filesystem", account)
store_backend = StoreBackend(namespace=create_name_space)


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
    backend=store_backend
)

@dataclass
class User:
    account:str
    password:str


# 第一次执行
config_1 = {"configurable":{"thread_id":"007"}}
result = deep_agent.invoke(input={
    "messages":[
        HumanMessage(content="我的名字叫大风哥,年龄是23岁,我喜欢38号!")
    ]
},config=config_1,context=User(account="root",password="123456"))  # zhangsan
print(f"第一次执行结果:{result['messages'][-1].content}")


# 我再执行一次!
config_2 = {"configurable":{"thread_id":"008"}}
result_2 = deep_agent.invoke(input={
    "messages":[
        HumanMessage(content="我是谁?我今年多大?我喜欢啥?")
    ]
    },config=config_2,
    # 必须传: 1) 不传则 runtime.context 为 None, 取 account 会报错
    #         2) namespace 按 account 隔离, 传同一个 account 才能读到第一次写入的数据
    context=User(account="root",password="123456")
)
print(f"第二次执行结果:{result_2['messages'][-1].content}")


# 确定有没有存进去! store
store.put(namespace=("qita","haha"),key="name",value="erdaye")
print(store.get(namespace=("qita","haha"),key="name"))

items = store.search(("filesystem","root"))
for item in items:
    print(item)
    print()
    print(item.key)
    print()
    print(item.value)