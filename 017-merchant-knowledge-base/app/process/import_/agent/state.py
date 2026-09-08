import copy
from typing import TypedDict



class ImportGraphState(TypedDict):

    task_id: str # 任务唯一ID, 用于追踪日志

    # 流程控制标记
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # 路径相关
    local_dir: str
    input_file_path: str     # 原始输入文件路径
    file_title: str          # 文件标题(文件名去后缀     )
    pdf_path: str            # PDF 文件路径
    md_path: str             # Markdown 文件路径

    # 内容数据
    md_content: str          # Markdown 全文内容
    chunks: list             # 分块后的文本列表，包含metadata
    item_name: str           # 识别出的主体名称（用于增强检索）
    

    # 数据库相关
    embeddings_content: list  # 包含向量数据的列表，准备写入Mivus



# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_dir": "",       # 本地输出目录
    "input_file_path": "", # 原始输入文件路径
    "file_title": "", 
    "pdf_path": "",
    "md_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}


def create_default_state(**kwargs) -> ImportGraphState:
    
    state = copy.deepcopy(graph_default_state)
    state.update(kwargs)

    return state

def get_default_state() -> ImportGraphState:
    return copy.deepcopy(graph_default_state)
