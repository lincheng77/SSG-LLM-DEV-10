import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """过滤与问题无关的表"""

    runtime.stream_writer({"stage": "过滤表"})

    query = state["query"]
    table_info_states = state["table_info_states"]

    # 1. 利用大模型过滤掉不需要的表和字段
    prompt_template = PromptTemplate(
        template=load_prompt("filter_table_info"),
        input_variables=["query", "table_infos"]
    )

    output_parser = JsonOutputParser()
    chain = prompt_template | llm | output_parser
    result = await chain.ainvoke({"query": query, "table_infos": yaml.dump(table_info_states, allow_unicode=True, sort_keys=False)})

    # 2. 删除大模型没有返回的表名、列名（遍历副本[:]，防止遍历的时候删除同一个列表）
    for table_info_state in table_info_states[:]:
        table_name = table_info_state["name"]
        if table_name not in result:
            table_info_states.remove(table_info_state)
        else:
            columns = table_info_state["columns"]
            for column in columns[:]:
                column_name = column["name"]
                if column_name not in result[table_name]:
                    columns.remove(column)


    logger.info(f"过滤表完成：{table_info_states}")

    return {"table_info_states": table_info_states}
