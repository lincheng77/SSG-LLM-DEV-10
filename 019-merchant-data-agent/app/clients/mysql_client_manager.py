import asyncio
from typing import Optional

from sqlalchemy import text, Select

from app.conf.app_config import DBConfig, app_config
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from app.models.mysql.table_info_mysql import TableInfoMySQL


class MysqlClientManager:
    """
    操作Mysql数据的客户端管理器类
    """

    def __init__(self, config: DBConfig):
        self.config = config

        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None

    def _get_url(self):
        return (f"mysql+asyncmy://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}?charset=utf8mb4")

    def init_client(self):
        # 创建引擎
        self.engine = create_async_engine(
            self._get_url(),
            pool_size=10,
            max_overflow=20, # 最大连接数
            pool_pre_ping=True, # 连接池是否在每次获取连接时检查连接是否有效
        )

        # 创建会话工厂
        self.session_factory = async_sessionmaker(
            self.engine,
            autobegin=True, # 自动开启事务
            expire_on_commit=False, # 事务提交后，ORM对象是否过期，为False代表不过期，还可以使用
            autoflush=True, # 在查询前是否自动将未提交事务的数据更新同步到暂存区 =》是否可以立即查询到  False查不到
        )
        
    async def close(self):
        if self.engine:
            await self.engine.dispose()
            self.engine = None


dw_mysql_client_manager = MysqlClientManager(app_config.db_dw)
meta_mysql_client_manager = MysqlClientManager(app_config.db_meta)
