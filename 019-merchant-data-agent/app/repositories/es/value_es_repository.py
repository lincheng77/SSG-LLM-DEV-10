import keyword

from elasticsearch import AsyncElasticsearch

from app.models.es.value_info_es import ValueInfoES


class ValueESRepository:

    index_name = "data-agent-value-index"

    mappings = {
        "dynamic": False, # 声明: 只接受下面定义好的字段, 写进来的文档若带未定义字段 → 忽略(不报错不索引)
        "properties": {
            "id": {"type": "keyword"},              # 精确值类型: 整个值当一个词条, 不分词, 只能精确匹配/过滤/聚合
            "value": {
                "type": "text",                     # 全文文本类型: 会被分词器切碎建立倒排索引, 支持模糊搜索
                "analyzer": "ik_max_word",          # 写入时分词器 中文分词器, 最细粒度切词 例: "用户消费金额" → [用户, 消费, 金额, 消费金额, ...]
                "search_analyzer": "ik_max_word"    # 搜索时分词器
            },
            "type": {"type": "keyword"},              # 精确值类型: 整个值当一个词条, 不分词, 只能精确匹配/过滤/聚合
            "column_id": {"type": "keyword"},
            "column_name": {"type": "keyword"},
            "table_id": {"type": "keyword"},
            "table_name": {"type": "keyword"},
        }
    }

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    # 确保索引存在
    async def _ensure_index(self):
        client = self.client
        index_name = self.index_name

        if await client.indices.exists(index=index_name):
            await client.indices.delete(index=index_name)

        await client.indices.create(
            index=index_name,
            mappings=self.mappings
        )

    # 批量插入多个字段值信息数据
    async def insert_values(self, values: list[ValueInfoES]):

        await self._ensure_index()

        index_dict = {
            "index":{
                "_index": self.index_name,
            }
        }

        # 将要保存的数据全部收集到operations中
        operations = []
        for value in values:
            operations.append(index_dict)
            operations.append(value)

        # 分批批量插入
        batch_size = 10
        for i in range(0, len(operations), batch_size):
            batch_operations = operations[i:i+batch_size]
            await self.client.bulk(operations=batch_operations)


    async def search(self, keyword: str) -> list[ValueInfoES]:
        result = await self.client.search(
            index = self.index_name,
            query={
                "match": {
                    "value": keyword
                }
            }
        )

        return [ValueInfoES(**item["_source"]) for item in result["hits"]["hits"]]
