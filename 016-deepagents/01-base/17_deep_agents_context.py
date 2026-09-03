from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from tavily import TavilyClient
from deepagents import create_deep_agent,FilesystemPermission
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


root_dir_path:Path = Path(__file__).parent # base
file_backend = FilesystemBackend(root_dir=root_dir_path,virtual_mode=True)


deep_agent = create_deep_agent(
    model = model,
    system_prompt=(
        "你是一名高情商中文助手。"
        "你的回答要自然、细腻、有陪伴感。"
        "当任务适合拆分时，可以交给子代理处理。"
        "如果用户要求写文案、祝福语、安慰话术，交给对应的技能blessing-writer实现! "
        "不要反问,一次完成输出!!"
    ),
    backend=file_backend,
    skills=[
        "/skills/"  # 技能的文件夹  file_backend / -> 之后写
    ],
    memory=["/project/AGENTS.md","/project/preferences.md"]
)


result1 = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写一份中秋节节日祝福!"}]},
    config={"configurable": {"thread_id": "demo_1"}}
)

print(f"[Agent]: {result1['messages'][-1].content}")

