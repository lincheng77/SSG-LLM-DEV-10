from minio import Minio

from app.infra.providers import infra_config
from app.shared.clients import get_minio_client


class MinioGateway:

    @property
    def bucket_name(self) -> str:
        """获取 MinIO 存储桶名称（从全局配置读取）"""
        return infra_config.minio.bucket_name

    @property
    def image_dir(self) -> str:
        """获取 MinIO 中存放图片的目录路径（从全局配置读取）"""
        return infra_config.minio.minio_img_dir



    def client(self) -> Minio:
        return get_minio_client()



    def build_image_url(self, stem:str, image_name:str) -> str:
        protocol = "https" if infra_config.minio.minio_secure else "http"

        return f"{protocol}://{infra_config.minio.endpoint}/{self.bucket_name}{self.image_dir}/{stem}/{image_name}"

# 创建全局唯一的 MinIO 网关实例，全项目复用
minio_gateway = MinioGateway()
