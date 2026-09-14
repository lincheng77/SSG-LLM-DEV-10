import json

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from app.agent.context import DataAgentContext
from app.agent.graph import graph
from app.agent.state import DataAgentState
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class QueryService:

    def __init__(self,
                 dw_mysql_repo: DWMysqlRepository,
                 meta_mysql_repo: MetaMysqlRepository,
                 value_es_repo: ValueESRepository,
                 column_qdrant_repo: ColumnQdrantRepository,
                 metric_qdrant_repo: MetricQdrantRepository,
                 embedding_client: HuggingFaceEndpointEmbeddings):
        self.dw_mysql_repo = dw_mysql_repo
        self.meta_mysql_repo = meta_mysql_repo
        self.value_es_repo = value_es_repo
        self.column_qdrant_repo = column_qdrant_repo
        self.metric_qdrant_repo = metric_qdrant_repo
        self.embedding_client = embedding_client

    async def search(self, query: str):
        state = DataAgentState(query=query)
        context = DataAgentContext(
            embedding_client=self.embedding_client,  # 向量服务器客户端
            column_qdrant_repository=self.column_qdrant_repo,  # 字段向量库
            metric_qdrant_repository=self.metric_qdrant_repo,  # 指标向量库
            value_es_repository=self.value_es_repo,  # 字段值全文索引库
            meta_mysql_repository=self.meta_mysql_repo,  # 元数据库
            dw_mysql_repository=self.dw_mysql_repo,  # 数据仓库
        )

        async for chunk in graph.astream(input=state, context=context, stream_mode="custom"):
            yield f"data:{json.dumps(chunk, ensure_ascii=False, default=str)}\n\n"
