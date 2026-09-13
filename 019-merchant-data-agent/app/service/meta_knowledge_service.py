import uuid

from app.conf.meta_config import MetaConfig, TableConfig
from app.core.log import logger
from app.models.es.value_info_es import ValueInfoES
from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository
from app.repositories.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from langchain_huggingface import HuggingFaceEndpointEmbeddings


class MetaKnowledgeService:

    def __init__(self,
                 dw_mysql_repository: DWMysqlRepository,
                 meta_mysql_repository: MetaMysqlRepository,
                 value_es_repository: ValueESRepository,
                 column_qdrant_repository: ColumnQdrantRepository,
                 metric_qdrant_repository: MetricQdrantRepository,
                 embedding_client: HuggingFaceEndpointEmbeddings):

        self.dw_mysql_repository = dw_mysql_repository
        self.meta_mysql_repository = meta_mysql_repository
        self.value_es_repository = value_es_repository
        self.column_qdrant_repository = column_qdrant_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.embedding_client = embedding_client

    async def build(self, config: MetaConfig):
        logger.info("开始构建业务")

        # 1. 处理表相关信息
        if config.tables:
            # 1.1 将表的信息和字段信息保存到meta库（table_info 和 column_info 表）
            column_infos: list[ColumnInfoMySQL] = await self._save_table_infos_to_meta_db(config.tables)

            # 1.2 对所有字段信息建立向量索引（表信息会打上字段的顺风车）
            await self._save_column_infos_to_qdrant(column_infos)

            # 1.3 对象字段取值建立全文索引
            await self._save_value_infos_to_es(column_infos, config.tables)

        # 2. 处理指标信息
        if config.metrics:
            # 2.1 将指标信息数据保存到meta库（metric_info 和 column_metric 表)
            metric_infos: list[MetricInfoMySQL] = await self._save_metric_infos_to_meta_db(config.metrics)

            # 2.2 对所有指标信息建立向量索引
            await self._save_metric_infos_to_qdrant(metric_infos)

    async def _save_table_infos_to_meta_db(self, tables) -> list[ColumnInfoMySQL]:
        """
          保存表的元数据和列的元数据信息 到元数据库（meta库）
        :param tables:
        :return:
        """
        # 准备容器
        table_infos: list[TableInfoMySQL] = []
        column_infos: list[ColumnInfoMySQL] = []

        # 准备表的元数据
        for table in tables:
            table_info = TableInfoMySQL(
                id=table.name,
                name=table.name,
                role=table.role,
                description=table.description
            )
            table_infos.append(table_info)

            # 准备列的元数据
            column_type_dict: dict[str, str] = await self.dw_mysql_repository.get_column_types(table.name)
            for column in table.columns:
                examples: list[dict[str, str]] = await self.dw_mysql_repository.get_column_values(table.name,
                                                                                                  column.name)

                column_info = ColumnInfoMySQL(
                    id=f"{table.name}.{column.name}",
                    name=column.name,
                    type=column_type_dict[column.name],
                    role=column.role,
                    examples=examples,
                    description=column.description,
                    alias=column.alias,
                    table_id=table_info.id
                )
                column_infos.append(column_info)

        # 保存到数据库
        await self.meta_mysql_repository.save_table_infos(table_infos)
        await self.meta_mysql_repository.save_column_infos(column_infos)

        return column_infos

    async def _save_column_infos_to_qdrant(self, column_infos: list[ColumnInfoMySQL]):
        """
          保存字段的元数据 到元数据库（meta库）
        :param column_infos:
        :return:
        """

        # 待向量化条目: 每项 = 一个搜索入口(字段名/描述/别名) + uuid + 归属的payload
        embedding_entries: list[dict] = []

        for column_info in column_infos:
            column_info_qdrant = ColumnInfoQdrant(
                id=column_info.id,
                name=column_info.name,
                type=column_info.type,
                role=column_info.role,
                examples=column_info.examples,
                description=column_info.description,
                alias=column_info.alias,
                table_id=column_info.table_id
            )

            # 同一个payload挂多个搜索入口: 字段名一条、描述一条、每个别名各一条
            embedding_entries.append({
                "id": uuid.uuid4(),
                "embedding_text": column_info.name,
                "payload": column_info_qdrant
            })
            embedding_entries.append({
                "id": uuid.uuid4(),
                "embedding_text": column_info.description,
                "payload": column_info_qdrant
            })
            for alia in column_info.alias:
                embedding_entries.append({
                    "id": uuid.uuid4(),
                    "embedding_text": alia,
                    "payload": column_info_qdrant
                })

        embedding_texts = [entry["embedding_text"] for entry in embedding_entries]

        batch_size = 32
        vectors: list[list[float]] = []
        for i in range(0, len(embedding_texts), batch_size):
            batch_embedding_texts = embedding_texts[i:i+batch_size]
            batch_vectors = await self.embedding_client.aembed_documents(batch_embedding_texts)
            vectors.extend(batch_vectors)


        ids: list[str] = [entry["id"] for entry in embedding_entries]
        payloads: list[dict] = [entry["payload"] for entry in embedding_entries]

        await self.column_qdrant_repository.upsert_column_vectors(vectors, payloads, ids)






    async def _save_value_infos_to_es(self, column_infos: list[ColumnInfoMySQL], tables: list[TableConfig]):
        """
          保存对象字段取值到全文索引（elasticsearch）
        :param column_infos:
        :return:
        """

        # 1. 准备对象字段取值的索引( 存储所有列的sync字段 )
        column_sync_dict: dict[str, str] = {}
        for table in tables:
            for column in table.columns:
                column_sync_dict[f"{table.name}.{column.name}"] = column.sync

        # 2. 准备对象字段取值的索引( 存储所有对象字段取值 )
        value_infos: list[ValueInfoES] = []
        for column_info in column_infos:
            sync = column_sync_dict[column_info.id]
            if sync:

                # 查询dw库得到该字段的所有值，对每个值，创建一个ValueInfoEs对象（）
                values = await self.dw_mysql_repository.get_column_values(column_info.table_id, column_info.name, 10000)
                for value in values:
                    value_info_es = ValueInfoES(
                        id=f"{column_info.id}.{value}",
                        value=value,
                        column_id=column_info.id,
                        table_id=column_info.table_id,
                    )

                    value_infos.append(value_info_es)

        # 3. 调用持久层保存到数据库ES中
        await self.value_es_repository.insert_values(value_infos)

    async def _save_metric_infos_to_meta_db(self, metrics):

        metric_infos : list[MetricInfoMySQL] = []
        column_metric: list[ColumnMetricMySQL] = []

        for metric in metrics:
            metric_infos.append(MetricInfoMySQL(
                id=metric.name,
                name=metric.name,
                description = metric.description,
                relevant_columns = metric.relevant_columns,
                alias = metric.alias
            ))

            for column_id in metric.relevant_columns:
                column_metric.append(ColumnMetricMySQL(
                    column_id = column_id,
                    metric_id = metric.name
                ))

        await self.meta_mysql_repository.save_metric_infos(metric_infos)
        await self.meta_mysql_repository.save_column_metrics(column_metric)

        return metric_infos

    async def _save_metric_infos_to_qdrant(self, metric_infos):
        embedding_entries: list[dict] = []

        for metric_info in metric_infos:
            metric_info_qdrant = MetricInfoQdrant(
                id=metric_info.id,
                name=metric_info.name,
                description=metric_info.description,
                alias=metric_info.alias,
                relevant_columns = metric_info.relevant_columns
            )

            embedding_entries.append({
                "id": uuid.uuid4(),
                "embedding_text": metric_info.name,
                "payload": metric_info_qdrant
            })
            embedding_entries.append({
                "id": uuid.uuid4(),
                "embedding_text": metric_info.description,
                "payload": metric_info_qdrant
            })
            for alia in metric_info.alias:
                embedding_entries.append({
                    "id": uuid.uuid4(),
                    "embedding_text": alia,
                    "payload": metric_info_qdrant
                })


        embedding_texts = [entry["embedding_text"] for entry in embedding_entries]
        batch_size = 32
        vectors: list[list[float]] = []
        for i in range(0, len(embedding_texts), batch_size):
            batch_embedding_texts = embedding_texts[i:i+batch_size]
            batch_vectors = await self.embedding_client.aembed_documents(batch_embedding_texts)
            vectors.extend(batch_vectors)


        ids: list[str] = [entry["id"] for entry in embedding_entries]
        payloads: list[dict] = [entry["payload"] for entry in embedding_entries]

        await self.metric_qdrant_repository.upsert_metric_vectors(vectors, payloads, ids)
