from dataclasses import dataclass

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





root_dir_path:Path = Path(__file__).parent # base
file_backend = FilesystemBackend(root_dir=root_dir_path,virtual_mode=True)

deep_agent = create_deep_agent(
    model = model,
    system_prompt="""
      你是一个综合助手! 可以使用自己的技能(skill)为用户提供技术支撑!
    """,
    backend=file_backend,
    skills=[
        "/skills/"  # 技能的文件夹  file_backend / -> 之后写
    ]
)


result = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="帮我查看下,当前agent有哪些技能skill,给我罗列出来!!")
        ]
    }
)
print(f"查询结果:{result['messages'][-1].content}")


result1 = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="帮我使用算命技能,算一下高圆圆和赵又廷啥时候离婚!")
        ]
    }
)
print(f"使用算命技能结果:{result1['messages'][-1].content}")


query1 = "我早上起床晚了，赶公交车差点摔倒，还好最后到了公司。使用emoji-translator技能翻译！"
result2 = deep_agent.invoke(
    {"messages": [{"role": "user", "content": query1}]}
)
print(f"使用翻译技能结果: {result2['messages'][-1].content}")




