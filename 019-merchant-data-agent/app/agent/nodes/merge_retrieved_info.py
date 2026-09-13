from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def merge_retrieved_info(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """汇总字段、取值、指标三路召回结果"""
    return {}
