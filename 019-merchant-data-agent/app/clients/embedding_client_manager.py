import asyncio

from typing import Optional
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    """
    嵌入式向量嵌入客户端管理器类
    用于管理 HuggingFaceEndpointEmbeddings 客户端的初始化和配置
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: Optional[HuggingFaceEndpointEmbeddings] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init_client(self):
        """
        初始化 HuggingFaceEndpointEmbeddings 客户端
        将拼接好的服务 URL 作为模型地址传入客户端
        """
        self.client = HuggingFaceEndpointEmbeddings(model=self._get_url())

    async def close(self):
        if self.client:
            await self.client.async_client.close()
            self.client = None


# 创建向量数据库客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)
