# TODO: 从参考项目拷贝, 尚未审阅理解, 待精读

# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION = "vlm"

# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 -> 一个pdf 约等于 1秒
MINERU_POLL_TIMEOUT_SECONDS = 600

# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3

# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30


# """
# 导入链配置模块，负责集中管理切块、主体识别、图片处理、向量化与入库相关策略参数。
# """


# CHUNK_MAX_SIZE = 1000
# CHUNK_SIZE = 600
# CHUNK_OVERLAP = 20

# ITEM_NAME_CONTEXT_CHUNK_K = 5
# ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 10000

# EMBEDDING_BATCH_SIZE = 5
# SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# MINERU_MODEL_VERSION = "vlm"
# MINERU_POLL_TIMEOUT_SECONDS = 600
# MINERU_POLL_INTERVAL_SECONDS = 3
# MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30

# MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512
# MILVUS_CHUNK_CONTENT_MAX_LENGTH = 65535
# MILVUS_VECTOR_DIM = 1024
