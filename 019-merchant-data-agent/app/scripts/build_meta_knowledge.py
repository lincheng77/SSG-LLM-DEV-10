import asyncio

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

async def start_build():
    # 1. 初始化所有客户端管理
    embedding_client_manager.init_client()
    es_client_manager.init_client()
    qdrant_client_manager.init_client()
    dw_mysql_client_manager.init_client()
    meta_mysql_client_manager.init_client()

    try:
        # 2. 创建构建业务的对象
        async with (dw_mysql_client_manager.session_factory() as dw_session,
                    meta_mysql_client_manager.session_factory() as meta_session):
            service = MetaKnowledgeService(dw_mysql_repository=DWMysqlRepository(dw_session),
                                           meta_mysql_repository=MetaMysqlRepository(meta_session),
                                           value_es_repository=ValueESRepository(es_client_manager.client),
                                           column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
                                           metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
                                           embedding_client=embedding_client_manager.client
                                           )

            await service.build(meta_config)

            await dw_session.commit()
            await meta_session.commit()

            logger.info("build meta knowledge success")
    finally:
        # 无论成功失败，都关闭异步HTTP客户端和连接池，避免 Unclosed client session 警告
        await embedding_client_manager.close()
        await es_client_manager.close()
        await qdrant_client_manager.close()
        await dw_mysql_client_manager.close()
        await meta_mysql_client_manager.close()


if __name__ == '__main__':
    asyncio.run(start_build())
