"""
模型提供者模块，统一封装聊天模型、Embedding 与 Reranker 的访问方式。
"""
from langchain_openai import ChatOpenAI

from app.infra.providers import infra_config
from app.shared.model import generate_embeddings, get_bge_m3_ef, get_llm_client, get_reranker_model


class LLMProvider:

    def chat(self, model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
        return get_llm_client(model=model, json_mode=json_mode)

    def vision_chat(self) -> ChatOpenAI:
        return get_llm_client(model=infra_config.llm.vl_model)


llm_provider = LLMProvider()
