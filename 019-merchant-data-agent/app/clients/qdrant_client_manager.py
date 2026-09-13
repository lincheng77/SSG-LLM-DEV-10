import asyncio
from random import random
from typing import Optional
from qdrant_client import AsyncQdrantClient, models
from app.conf.app_config import QdrantConfig, app_config

class QdrantClientManager:
    """
    操作Qdrant数据的客户端管理器类
    """

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init_client(self):
        self.client = AsyncQdrantClient(
            url=self._get_url())
    
    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None


# 创建向量数据库客户端管理器
qdrant_client_manager = QdrantClientManager(app_config.qdrant)
