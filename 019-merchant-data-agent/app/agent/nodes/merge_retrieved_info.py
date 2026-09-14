from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, TableInfoState, ColumnInfoState, MetricInfoState
from app.core.log import logger
from app.models.es import value_info_es
from app.models.es.value_info_es import ValueInfoES
from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant





async def merge_retrieved_info(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """汇总字段、取值、指标三路召回结果"""
    runtime.stream_writer({"stage": "合并召回"})

    recall_columns: list[ColumnInfoQdrant] = state["recall_columns"]
    recall_metrics: list[MetricInfoQdrant] = state["recall_metrics"]
    recall_values: list[ValueInfoES] = state["recall_values"]
    meta_mysql_repository = runtime.context["meta_mysql_repository"]


    # 1. 把召回指标 的 【相关列】 加入到 召回列
    column_infos_dict: dict[str, ColumnInfoQdrant] = {item["id"]: item for item in recall_columns}
    for metric in recall_metrics:
        for relevant_column_id in metric["relevant_columns"]:
            if relevant_column_id not in column_infos_dict:
                column_info_mysql: ColumnInfoMySQL = await meta_mysql_repository.get_column_info_by_id(relevant_column_id)
                if column_info_mysql is None:
                    logger.warning(f"指标关联的字段在meta库中不存在，已跳过：{relevant_column_id}")
                    continue
                column_infos_dict[relevant_column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)

    # 2. 把全文召回的值 的 【所属列】 加入到 召回列
    for recall_value in recall_values:
        column_id = recall_value["column_id"]
        if column_id not in column_infos_dict:
            column_info_mysql: ColumnInfoMySQL = await meta_mysql_repository.get_column_info_by_id(column_id)
            if column_info_mysql is None:
                logger.warning(f"召回值关联的字段在meta库中不存在，已跳过：{column_id}")
                continue
            column_infos_dict[column_id] = _convert_column_info_mysql_to_qdrant(column_info_mysql)

        # 如果全文召回的值 还不是对应列的示例值， 可以加进去
        if recall_value["value"] not in column_infos_dict[column_id]["examples"]:
            column_infos_dict[column_id]["examples"].append(recall_value["value"])


    # 3. 根据召回的列， 【所属表的主外键】 加入到 召回列
    table_column_infos_dict: dict[ str, list[ColumnInfoQdrant]] ={}
    for column_info in column_infos_dict.values():
        table_id = column_info["table_id"]
        if table_id not in table_column_infos_dict:
            table_column_infos_dict[table_id] = []
        table_column_infos_dict[table_id].append(column_info)

    for table_id in table_column_infos_dict.keys():
        key_and_foreign_key_column: list[ColumnInfoMySQL] = await meta_mysql_repository.get_key_foreign_by_table_id(table_id)
        for column_info_mysql in key_and_foreign_key_column:
            if column_info_mysql.id not in column_infos_dict:
                column_infos_dict[column_info_mysql.id] = _convert_column_info_mysql_to_qdrant(column_info_mysql) #这个其实后面用不到了，加上也没事，也不会返回更新state
                table_column_infos_dict[table_id].append(column_infos_dict[column_info_mysql.id])

    # 4. 根据召回的列，封装对应的表信息(表里封装列信息) state
    table_info_state: list[TableInfoState] = []
    column_info_states: list[ColumnInfoState] = []
    for table_id, column_infos in table_column_infos_dict.items():
        table_info_mysql: TableInfoMySQL = await meta_mysql_repository.get_table_info_by_table_id(table_id)
        column_info_states: list[ColumnInfoState] = [_convert_column_info_qdrant_to_state(item) for item in column_infos]
        table_info_state.append(_convert_table_info_mysql_to_state(table_info_mysql, column_info_states))

    # 5. 指标信息转换
    metric_info_state = [_convert_metric_info_qdrant_to_state(item) for item in recall_metrics]

    logger.info(f"合并表信息列表完成： {table_info_state}")
    logger.info(f"合并列信息列表完成： {column_info_states}")
    logger.info(f"合并处理指标列表完成：{metric_info_state}")

    return {"table_info_states": table_info_state, "metric_info_states": metric_info_state}

def _convert_column_info_mysql_to_qdrant(column_info_mysql):
    return ColumnInfoQdrant(
        id=column_info_mysql.id,
        name=column_info_mysql.name,
        description=column_info_mysql.description,
        role=column_info_mysql.role,
        type=column_info_mysql.type,
        examples=column_info_mysql.examples,
        table_id=column_info_mysql.table_id,
        alias=column_info_mysql.alias
    )

def _convert_column_info_qdrant_to_state(item):
    return ColumnInfoState(
        name=item["name"],
        type=item["type"],
        role=item["role"],
        examples=item["examples"],
        description=item["description"],
        alias=item["alias"]
    )


def _convert_table_info_mysql_to_state(table_info_mysql, column_info_states):
    return TableInfoState(
        name=table_info_mysql.name,
        role=table_info_mysql.role,
        description=table_info_mysql.description,
        columns=column_info_states
    )


def _convert_metric_info_qdrant_to_state(item):
    return MetricInfoState(
        name=item["name"],
        description=item["description"],
        relevant_columns=item["relevant_columns"],
        alias=item["alias"]
    )
