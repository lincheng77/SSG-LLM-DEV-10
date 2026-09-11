from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import app_config
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant


class MetricQdrantRepository:
    collection_name = "data-agent-metric-collection"

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def _ensure_collection(self):
        client = self.client
        collection_name = self.collection_name

        # 确保集合存在：如果存在，先删除，后面创建
        if await client.collection_exists(collection_name=collection_name):
            await client.delete_collection(collection_name=collection_name)
        await client.create_collection(
            collection_name=collection_name,  # 集合名称
            vectors_config=models.VectorParams(
                size=app_config.qdrant.embedding_size,  # 向量的维度
                distance=models.Distance.COSINE  # 余弦相似匹配
            ),
        )



    async def upsert_metric_vectors(self,
                                    vectors: list[list[float]],
                                    payloads: list[MetricInfoQdrant],
                                    ids: list[str]):
        """
        TODO

        :param vectors:
        :param payloads:
        :param ids: 向量的主键，来源是指标的业务主键转成的 UUID，作用是让重复导入变成覆盖而不是堆积
        :return:
        """
        client = self.client
        collection_name = self.collection_name

        await  self._ensure_collection()

        batch_size = 10
        for i in range(0, len(vectors), batch_size):
            # 当前批次的数据
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # 插入数据
            await client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=batch_ids[j],
                        vector=batch_vectors[j],
                        payload=batch_payloads[j]) for j in range(len(batch_vectors))
                ]
            )
