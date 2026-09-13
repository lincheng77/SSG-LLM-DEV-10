
from langgraph.constants import START,END
from langgraph.graph import StateGraph

from app.agent.context import DataAgentContext
from app.agent.nodes import (
    add_extra_context,
    correct_sql,
    execute_sql,
    extract_keywords,
    filter_metric,
    filter_table,
    generate_sql,
    merge_retrieved_info,
    recall_column,
    recall_metric,
    recall_value,
    validate_sql,
)
from app.agent.state import DataAgentState

graph = StateGraph(
    state_schema=DataAgentState,
    context_schema=DataAgentContext
)

# 添加节点
graph.add_node("extract_keywords", extract_keywords)
graph.add_node("recall_column", recall_column)
graph.add_node("recall_value", recall_value)
graph.add_node("recall_metric", recall_metric)

graph.add_node("merge_retrieved_info", merge_retrieved_info)
graph.add_node("filter_metric", filter_metric)
graph.add_node("filter_table", filter_table)
graph.add_node("add_extra_context", add_extra_context)
graph.add_node("generate_sql", generate_sql)
graph.add_node("validate_sql", validate_sql)
graph.add_node("correct_sql", correct_sql)
graph.add_node("execute_sql", execute_sql)

# 添加边
graph.add_edge(START, "extract_keywords")

graph.add_edge("extract_keywords", "recall_column")
graph.add_edge("extract_keywords", "recall_metric")
graph.add_edge("extract_keywords", "recall_value")

graph.add_edge("recall_column", "merge_retrieved_info")
graph.add_edge("recall_metric", "merge_retrieved_info")
graph.add_edge("recall_value", "merge_retrieved_info")

graph.add_edge("merge_retrieved_info", "filter_metric")
graph.add_edge("merge_retrieved_info", "filter_table")

graph.add_edge("filter_metric", "add_extra_context")
graph.add_edge("filter_table", "add_extra_context")

graph.add_edge("add_extra_context", "generate_sql")
graph.add_edge("generate_sql", "validate_sql")
# 添加条件边
graph.add_conditional_edges(
    "validate_sql",
    lambda state: "execute_sql" if state.get("error") is None else "correct_sql",
    {
        "execute_sql": "execute_sql",
        "correct_sql": "correct_sql"}
)
graph.add_edge("correct_sql", "execute_sql")
graph.add_edge("execute_sql", END)



# 编译
compiled_graph = graph.compile()
