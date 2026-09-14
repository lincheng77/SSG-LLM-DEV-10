from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
import yaml
from app.agent.llm import llm
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.prompt.prompt_loader import load_prompt
from app.core.log import logger
from langchain_core.output_parsers import StrOutputParser


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """根据校验错误信息修正SQL"""
    runtime.stream_writer({"stage": "根据校验错误信息修正SQL"})

    query = state["query"]
    table_info_states = state["table_info_states"]
    metric_info_states = state["metric_info_states"]
    date_info = state["date_info"]
    db_info = state["db_info"]
    sql = state["sql"]
    error = state["error"]

    # 1. 利用大模型纠正SQL
    prompt_template = PromptTemplate(
        template=load_prompt("correct_sql"),
        input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info", "sql", "error"]
    )

    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    result = await chain.ainvoke({
        "query": query,
        "table_infos": yaml.dump(table_info_states, allow_unicode=True, sort_keys=False),
        "metric_infos": yaml.dump(metric_info_states, allow_unicode=True, sort_keys=False),
        "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
        "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
        "sql": sql,
        "error": error
    })

    logger.info(f"修正的SQL：{result}")

    return {"sql": result}
