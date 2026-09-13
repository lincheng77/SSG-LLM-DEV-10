from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def add_extra_context(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """补充生成SQL所需的上下文信息"""
    return {}
