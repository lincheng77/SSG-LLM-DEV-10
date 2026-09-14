from typing import TypedDict

from app.models.es.value_info_es import ValueInfoES
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant


class ColumnInfoState(TypedDict):
    name: str
    type:str
    role: str
    examples:list
    description:str
    alias: list

class TableInfoState(TypedDict):
    name:str
    role:str
    description:str
    columns: list[ColumnInfoState]


class MetricInfoState(TypedDict):
    name:str
    description:str
    relevant_columns:list
    alias:list

class DataAgentState(TypedDict):
    query: str # 用户提问
    keywords: list[str] # 提问分词产生的多个关键字


    recall_columns: list[ColumnInfoQdrant] # 列信息对象列表
    recall_metrics: list[MetricInfoQdrant] # 指标信息对象列表
    recall_values: list[ValueInfoES] # 字段值信息对象列表


    table_info_states: list[TableInfoState]
    metric_info_states: list[MetricInfoState]
