import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """过滤与问题无关的指标"""
    runtime.stream_writer({"stage": "过滤指标"})

    query = state["query"]
    metric_info_states = state["metric_info_states"]

    # 1. 利用大模型过滤掉不需要的指标
    prompt_template = PromptTemplate(
        template=load_prompt("filter_metric_info"),
        input_variables=["query", "metric_infos"]
    )

    output_parser = JsonOutputParser()
    chain = prompt_template | llm | output_parser
    result = await chain.ainvoke({"query": query, "metric_infos": yaml.dump(metric_info_states, allow_unicode=True, sort_keys=False)})


    # 2. 对指标信息进行过滤
    for metric_info in metric_info_states[:]:
        metric_name = metric_info['name']
        if metric_name not in result:
            metric_info_states.remove(metric_info)


    logger.info(f"过滤指标完成：{metric_info_states}")
    return {"metric_info_states": metric_info_states}
