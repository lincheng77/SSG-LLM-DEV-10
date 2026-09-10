import asyncio

from typing import Optional
from elasticsearch import AsyncElasticsearch
from app.conf.app_config import ESConfig, app_config


class ESClientManager:
    """
    ES异步客户端管理器：封装ES客户端的初始化、连接、关闭逻辑
    """

    def __init__(self, config: ESConfig):
        self.config = config
        self.client: Optional[AsyncElasticsearch] = None

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init_client(self):
        self.client = AsyncElasticsearch(
            hosts=[self._get_url()]
        )

    async def close(self):
        await self.client.close()


# 创建ES客户端管理器对象
es_client_manager = ESClientManager(app_config.es)
