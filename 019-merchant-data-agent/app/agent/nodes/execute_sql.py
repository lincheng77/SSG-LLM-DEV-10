from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger

async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """在DW库执行SQL并保存查询结果"""
    runtime.stream_writer({"stage": "执行SQL"})

    sql = state["sql"]
    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    result= await dw_mysql_repository.execute_sql(sql)

    # 将查询结果写给调用者
    runtime.stream_writer({"result": result})

    logger.info(f"执行SQL完成 result={result}")

