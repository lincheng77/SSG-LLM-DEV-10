

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

import os
from deepagents import create_deep_agent,FilesystemPermission
from deepagents.backends import StateBackend,StoreBackend,FilesystemBackend,CompositeBackend
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


root_dir_path:Path = Path(__file__).parent / "agent_workspace"
file_backend = FilesystemBackend(root_dir=root_dir_path,virtual_mode=True)

deep_agent = create_deep_agent(
    model = model,
    system_prompt="""
      你是一个综合助手!
    """,
    backend=file_backend,
    permissions=[
        FilesystemPermission(
            operations=["write","read"],
            paths=["/agent_workspace/**"],
            mode="allow"
        )
        ,
        FilesystemPermission(
            operations=["write","read"],
            paths=["/**"],
            mode="deny"
        )
    ]
)

result1 = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="读取info.txt文件,大风哥今年多大了!")
        ]
    }
)
print(f"读取的结果:{result1['messages'][-1].content}")

result2 = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="把我爱学习!写入到/agent_workspace/python.md文件中!")
        ]
    }
)
print(f"写入的结果:{result2['messages'][-1].content}")
