from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os
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
# 显式传 store: 图内执行时 create_deep_agent(store=...) 已能注入, 但脚本末尾在图外
# 直接调 composite_backend.ls("/store/") 时, StoreBackend 取不到 store 会报 RuntimeError
store_backend = StoreBackend(store=store, namespace=lambda _:("filesystem",))

roo_dir_path:Path = Path(__file__).parent / "agent_workspace"
roo_dir_path.mkdir(parents=True,exist_ok=True)
file_backend = FilesystemBackend(root_dir=roo_dir_path,virtual_mode=True)

# 复合backend
composite_backend = CompositeBackend(
    default=file_backend,  # 默认 没有触发路径的固定使用默认
    routes={
        "/store/": store_backend
    }
)

# 创建deep_agent
deep_agent = create_deep_agent(
    model = model,
    tools=[],
    subagents=[],
    system_prompt="""
      你是一个综合助手!
    """,
    store=store,
    backend=composite_backend
)

config = {"configurable":{"thread_id":"007"}}
result1 = deep_agent.invoke(
    input={
        "messages":[
            HumanMessage(content="我的名字叫大风哥,年龄是23岁,我喜欢38号! 将我的信息存到 /store/info.txt文件")
        ]
    },config = config)
print(f"第一次执行结果:{result1['messages'][-1].content}")


result2 = deep_agent.invoke(
    input={
        "messages":[
            HumanMessage(content="我的名字叫驴蛋蛋,年龄是5岁,我喜欢吃骨头和实心肉! 将我的信息存到 /test/info.txt文件")
        ]
    },config = config)
print(f"第二次执行结果:{result2['messages'][-1].content}")



items = store.search(("filesystem",))
print(items)


