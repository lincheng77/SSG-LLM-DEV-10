from sqlalchemy import select, text
from sqlalchemy.ext.asyncio.session import AsyncSession

from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL


class DWMysqlRepository:
    """
    操作Meta Mysql数据的仓库类
    """
    def __init__(self, session: AsyncSession):
        self.session = session

    def save_table_infos(self, table_infos: list[TableInfoMySQL]):
        self.session.add_all(table_infos)

    def save_column_infos(self, column_infos: list[ColumnInfoMySQL]):
        self.session.add_all(column_infos)

    def save_metric_infos(self, metric_infos: list[MetricInfoMySQL]):
        self.session.add_all(metric_infos)

    def save_column_metrics(self, column_metrics: list[ColumnMetricMySQL]):
        self.session.add_all(column_metrics)

    # 根据字段id获取字段信息
    async def get_column_info_by_id(self, column_id: int):
        return await self.session.get(ColumnInfoMySQL, column_id)

    # 根据表id获取主键和外键的字段信息
    async def get_key_foreign_by_table_id(self, table_id: int) -> list[ColumnInfoMySQL]:
        result = await self.session.execute(
            select(ColumnInfoMySQL).where(ColumnInfoMySQL.table_id == table_id)
                .where(ColumnInfoMySQL.role.in_(['primary_key', 'foreign_key']))
        )
        return result.scalars().all()   #select(ColumnInfoMySQL) 查询出来的第一列就不是某个字段，而是整个实体


"""
.fetchall()  所有行的列表，每行是 Row（像元组，也能按列名取值）      [(1,'阿司匹林'), (2,'布洛芬'), (3,'感冒灵')]
.first()     只取第一行（Row），没查到返回 None                    (1,'阿司匹林')
.scalar()    第一行第一列的单个值，没查到返回 None                  1
.scalars()   每行的第一列组成的流，all() 收成列表                   [1,2,3]
.mappings()  每行转成字典的流，all() 收成字典列表                   [{'id':1,'name':'阿司匹林'}, {'id':2,'name':'布洛芬'}, ...]
"""