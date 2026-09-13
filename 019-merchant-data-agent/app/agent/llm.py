import os

from langchain.chat_models import init_chat_model

from app.conf.app_config import app_config

llm = init_chat_model(
    model=app_config.llm.model_name,
    api_key= app_config.llm.api_key,
    # base_url="",
    temperature=0 # 让大型处理的结果稳定，不要创意
)
