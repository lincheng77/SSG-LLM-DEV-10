from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 1. 加载配置文件
load_dotenv(override=True)
#2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 3. 创建基于langchain的agent
langchain_agent = create_agent(
    model=model,
    system_prompt="我们是一个算数助手,专门用于数学运算!"
)


#4. 将langchain的使用compiledSubagent包裹和编译即可
langchain_subagent = CompiledSubAgent(
    name="langchain_subagent",
    description="负责进行算数运算的子智能体,如有运算需求使用此智能体!",
    runnable=langchain_agent
)

# 5.多智能体
deep_agent = create_deep_agent(
    model=model,
    system_prompt="""
    """,
    tools=[],
    subagents=[langchain_subagent]
)