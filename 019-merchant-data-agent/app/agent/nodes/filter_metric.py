from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def filter_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """过滤与问题无关的指标"""
    return {}
