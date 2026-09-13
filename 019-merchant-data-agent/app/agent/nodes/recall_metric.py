from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.prompt.prompt_loader import load_prompt


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """根据关键词从Qdrant召回相关指标"""
    runtime.stream_writer({"stage": "召回指标"})


    query = state["query"]
    keywords = state["keywords"]

    embedding_client = runtime.context["embedding_client"]
    metric_qdrant_repository = runtime.context["metric_qdrant_repository"]


    # 1. 利用大模型进行语义化分词
    prompt_template = PromptTemplate(
        template=load_prompt("extend_keywords_for_metric_recall"),
        input_variables=["query"]
    )

    output_parser = JsonOutputParser()
    chain = prompt_template | llm | output_parser
    result = chain.invoke({"query": query})
    logger.info(f"大模型返回的语义化分词为：{result}")
    keywords = list(set(keywords + result))

    # 2. Qdrant 召回
    metric_infos_dict: dict[str, MetricInfoQdrant] = {}
    for keyword in keywords:
        vector = await embedding_client.aembed_query(keyword)
        metric_infos: list[MetricInfoQdrant] = await metric_qdrant_repository.search(vector)

        for metric_info in metric_infos:
            metric_id = metric_info["id"]
            if metric_id not in metric_infos_dict:
                metric_infos_dict[metric_id] = metric_info


    # 3. 生成目标数据结构
    recall_metrics: list[MetricInfoQdrant] = list(metric_infos_dict.values())
    logger.info(f"召回指标完成：{recall_metrics}")

    return {"recall_metrics": recall_metrics}
