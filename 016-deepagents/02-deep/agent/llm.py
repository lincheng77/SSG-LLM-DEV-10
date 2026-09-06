from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os


load_dotenv(override=True)

model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
