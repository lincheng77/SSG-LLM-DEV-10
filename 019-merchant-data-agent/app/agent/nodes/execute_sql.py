from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState


async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """在DW库执行SQL并保存查询结果"""
    return {}
