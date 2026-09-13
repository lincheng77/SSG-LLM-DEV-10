from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.models.es.value_info_es import ValueInfoES
from app.prompt.prompt_loader import load_prompt


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """根据关键词从ES召回相关字段取值"""
    runtime.stream_writer({"stage": "召回列的值"})


    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    value_es_repository = runtime.context["value_es_repository"]

    # 1. 利用大模型进行语义化分词
    prompt_template = PromptTemplate(
        template=load_prompt("extend_keywords_for_value_recall"),
        input_variables=["query"]
    )

    output_parser = JsonOutputParser()
    chain = prompt_template | llm | output_parser
    result = chain.invoke({"query": query})
    logger.info(f"大模型返回的语义化分词为：{result}")
    keywords = list(set(keywords + result))

    # 2. ES 全文召回
    value_infos_dict: dict[str, ValueInfoES] = {}
    for keyword in keywords:
        value_infos: list[ValueInfoES] = await value_es_repository.search(keyword)

        for value_info in value_infos:
            value_id = value_info["id"]
            if value_id not in value_infos_dict:
                value_infos_dict[value_id] = value_info

    # 3. 生成目标数据结构
    recall_values: list[ValueInfoES] = list(value_infos_dict.values())
    logger.info(f"召回列的值完成：{recall_values}")

    return {"recall_values": recall_values}
