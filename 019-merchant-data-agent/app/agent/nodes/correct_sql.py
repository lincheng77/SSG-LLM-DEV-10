from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """根据校验错误信息修正SQL"""
    return {}
