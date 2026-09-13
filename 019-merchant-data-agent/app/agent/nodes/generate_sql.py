from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """调用LLM根据上下文生成SQL"""
    return {}
