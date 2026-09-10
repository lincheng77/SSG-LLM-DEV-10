"""
工具模块，负责提供 minio 相关的辅助能力。
"""
import json

from minio import Minio

# 导入项目内部配置与日志工具
from app.shared.config.minio_config import minio_config  # MinIO相关配置（端点、密钥、桶名等）
from app.shared.runtime.logger import logger            # 项目统一日志工具

# 全局MinIO客户端实例（单例模式，避免重复创建连接，提升性能）
_minio_client: Minio | None = None


# 1. 定义MinIO客户端连接创建函数（私有函数，仅内部调用）
def _create_minio_client() -> Minio:
    return Minio(
        endpoint=minio_config.endpoint,        # MinIO服务端点（IP:端口）
        access_key=minio_config.access_key,    # MinIO访问密钥
        secret_key=minio_config.secret_key,    # MinIO秘密密钥
        secure=minio_config.minio_secure       # 是否启用HTTPS（True/False）
    )


# 2. 定义桶访问策略生成函数（私有函数，仅内部调用）
def _build_bucket_policy(bucket_name: str) -> str:
    # 策略模板（遵循AWS S3策略规范，MinIO兼容该规范）
    policy = {
        "Version": "2012-10-17",  # 策略版本（固定值，兼容S3标准）
        "Statement": [
            {
                "Effect": "Allow",  # 策略效果：允许访问
                "Principal": {"AWS": ["*"]},  # 授权对象：所有用户
                "Action": ["s3:GetObject"],  # 授权操作：读取桶内对象
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],  # 授权范围：桶内所有对象
            }
        ],
    }
    # 将字典策略序列化为JSON字符串，供MinIO设置使用
    return json.dumps(policy)


# 3. 定义桶初始化函数（私有函数，仅内部调用）
def _create_bucket_ready(client: Minio):
    bucket_name = minio_config.bucket_name  # 从配置中获取目标桶名
    # 检查桶是否存在
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)  
        client.set_bucket_policy(bucket_name, _build_bucket_policy(bucket_name))
        logger.info(f"MinIO桶 {bucket_name} 已创建，并设置访问策略")
    else:
        # 桶已存在，仅打印日志，不重复操作
        logger.info(f"MinIO桶 {bucket_name} 已存在，无需重复创建")


def get_minio_client() -> Minio:
    """
    :return: 全局唯一的MinIO客户端对象
    """
    # 声明使用全局变量（修改全局变量需显式声明）
    global _minio_client

    # 懒加载：仅在客户端未初始化时执行创建逻辑
    if _minio_client is None:
        logger.info("开始初始化MinIO客户端（首次调用，执行懒加载）")
        client = _create_minio_client()          # 创建客户端连接
        _create_bucket_ready(client)             # 检查并初始化桶
        _minio_client = client                   # 赋值给全局变量，供后续复用
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    # 复用全局客户端实例，直接返回
    return _minio_client
