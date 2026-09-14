from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.clients.ESClientManager import es_client_manager
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.core.log import logger
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.service.meta_knowledge_service import MetaKnowledgeService
from app.conf.meta_config import meta_config


@asynccontextmanager
async def lifespan(app: FastAPI):


    # 初始化所有客户端管理
    embedding_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()


    yield


    # 关闭所有客户端管理
    await es_client_manager.close()
    await qdrant_client_manager.close()
    await dw_mysql_client_manager.close()
    await meta_mysql_client_manager.close()
