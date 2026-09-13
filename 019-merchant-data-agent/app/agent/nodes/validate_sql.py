from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """校验SQL语法与表字段引用是否合法"""
    return {}
