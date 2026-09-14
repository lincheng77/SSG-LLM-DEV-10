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
        :param vectors:
        :param payloads:
        :param ids: 每个向量点的唯一编号(UUID, 由调用方随机生成), 与vectors/payloads按下标一一对应组装成点; 集合每次导入前整体删除重建(见_ensure_collection), 故id无需确定性
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


    async def search(self, vertor: list[float], score_threshold=0.6) -> list[MetricInfoQdrant]:
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=vertor,
            score_threshold=score_threshold
        )

        return [MetricInfoQdrant(**point.payload) for point in result.points]
