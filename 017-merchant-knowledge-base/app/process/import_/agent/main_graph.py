from langgraph.graph import StateGraph, END, START
from app.shared.runtime.logger import logger
from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus


# 1. 定义状态图对象，并制定全局state类型
workflow = StateGraph(ImportGraphState)

# 2. 添加节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding) # BGE向量化节点：将文本分块编码为向量，生成待写入Milvus的数据
workflow.add_node("node_import_milvus", node_import_milvus)


# 3. 设置入口节点
workflow.add_edge(START, "node_entry")

# 4. 设置条件边：根据文件类型路由到不同节点
def after_node_entry(state: ImportGraphState) -> str:
    if state.get("is_pdf_read_enabled", False):
        logger.info(f"传入的文件地址:{state.get('input_file_path')},文件类型为pdf,跳转到node_pdf_to_md节点!")
        return "node_pdf_to_md"
    if state.get("is_md_read_enabled", False):
        logger.info(f"传入的文件地址:{state.get('input_file_path')},文件类型为md,跳转到node_md_img节点!")
        return "node_md_img"
    else:
        logger.info(f"传入的文件地址:{state.get('input_file_path')},文件既不是md又不是pdf,无法处理,直接跳到END节点!")
        return END
workflow.add_conditional_edges("node_entry", after_node_entry, {
    "node_md_img": "node_md_img",    # 键 → 实际节点（可以不要，因为三个节点的键值一样）
    "node_pdf_to_md": "node_pdf_to_md", # 键 → 实际节点（可以不要，因为三个节点的键值一样）
    END: END,                                # 键 → 实际节点（可以不要，因为三个节点的键值一样）
}) 

# 5. 设置静态边：定义节点的执行顺序
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)
        
# 6. 编译状态图
bk_import_app = workflow.compile()