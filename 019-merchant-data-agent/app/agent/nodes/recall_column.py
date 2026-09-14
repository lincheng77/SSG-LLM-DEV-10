from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.llm import llm
from app.core.log import logger
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.prompt.prompt_loader import load_prompt
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """根据关键词从Qdrant召回相关字段"""
    runtime.stream_writer({"stage": "召回字段"})


    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    column_qdrant_repository = runtime.context["column_qdrant_repository"]


    # 1. 利用大模型进行语义化分词
    prompt_template = PromptTemplate(
        template=load_prompt("extend_keywords_for_column_recall"),
        input_variables=["query"]
    )

    output_parser = JsonOutputParser()
    chain = prompt_template | llm | output_parser
    result = await chain.ainvoke({"query": query})
    logger.info(f"大模型返回的语义化分词为：{result}")
    keywords = list(set(keywords + result))

    # 2. Qdrant 召回
    column_infos_dict: dict[str, ColumnInfoQdrant] = {}
    for keyword in keywords:
        vector = await embedding_client.aembed_query(keyword)
        column_infos: list[ColumnInfoQdrant] = await column_qdrant_repository.search(vector)

        for column_info in column_infos:
            column_id = column_info["id"]
            if column_id not in column_infos_dict:
                column_infos_dict[column_id] = column_info


    # 3. 生成目标数据结构
    recall_columns: list[ColumnInfoQdrant] = list(column_infos_dict.values())
    logger.info(f"召回字段完成：{recall_columns}")

    return {"recall_columns": recall_columns}
