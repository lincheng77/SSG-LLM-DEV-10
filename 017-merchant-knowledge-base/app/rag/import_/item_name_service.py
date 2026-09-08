# TODO: 从参考项目拷贝, 尚未审阅理解, 待精读

"""
主体识别服务模块，负责从导入文档中识别主体名称并写入主体索引。
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.rag.import_.config import (
    ITEM_NAME_CONTEXT_CHUNK_K,
    ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state: dict) -> tuple[list[dict], str]:
    chunks = state.get("chunks", [])
    file_title = state.get("file_title")
    if not chunks:
        logger.error("chunks没有内容,无法继续业务!")
        raise ValueError("chunks没有内容,无法继续业务!")
    if not file_title:
        logger.warning("file_title为空给与默认值处理!")
        file_title = "default_title"
        state["file_title"] = file_title
    return chunks, file_title


@step_log("build_document_context")
def build_document_context(chunks: list[dict]) -> str:
    current_chunks = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    chunk_str_list: list[str] = []
    for index, item in enumerate(current_chunks, start=1):
        chunk_str_list.append(f"切片:{index},标题:{item['title']},内容:{item['content']}")
    chunk_str = "\n".join(chunk_str_list)
    return chunk_str[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]


@step_log("recognize_item_name")
def recognize_item_name(context: str, file_title: str) -> str:
    llm = llm_provider.chat()
    system_prompt_str = load_prompt("product_recognition_system")
    user_prompt_str = load_prompt("item_name_recognition", file_title=file_title, context=context)
    messages = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=user_prompt_str),
    ]
    item_name = (llm | StrOutputParser()).invoke(messages)
    return item_name or file_title


@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str) -> list[dict]:
    for chunk in chunks:
        chunk["item_name"] = item_name
    return chunks


@step_log("embed_item_name")
def embed_item_name(item_name: str) -> tuple[list[float], dict[int, float]]:
    result = llm_provider.embed_documents([item_name])
    return result["dense"][0], result["sparse"][0]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection() -> None:
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.item_name_collection
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, dense_vector: list[float], sparse_vector: dict[int, float]) -> None:
    milvus_client = milvus_gateway.client()
    prepare_item_name_collection()
    safe_item_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=milvus_gateway.item_name_collection,
        filter=f"item_name == '{safe_item_name}'",
    )
    milvus_client.insert(
        collection_name=milvus_gateway.item_name_collection,
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
        ],
    )


@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: dict) -> dict:
    chunks, file_title = validate_chunks_and_title(state)
    # 从前若干个切片拼接上下文，给模型一个足够稳定的识别窗口。
    context = build_document_context(chunks)
    # 让模型输出当前文档对应的主体名，识别失败时会回退到文件标题。
    item_name = recognize_item_name(context, file_title)
    state["item_name"] = item_name
    state["chunks"] = apply_item_name(chunks, item_name)
    # 主体名本身也会生成向量，便于查询阶段做主体确认。
    dense_vector, sparse_vector = embed_item_name(item_name)
    upsert_item_name(item_name, file_title, dense_vector, sparse_vector)
    return state
