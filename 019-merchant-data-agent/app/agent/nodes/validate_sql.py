from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """校验SQL语法与表字段引用是否合法"""
    runtime.stream_writer({"stage": "校验SQL"})

    sql = state["sql"]
    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    error = await dw_mysql_repository.validate_sql(sql)

    # None 表示通过，条件边据此路由到 execute_sql / correct_sql
    return {"error": error}
