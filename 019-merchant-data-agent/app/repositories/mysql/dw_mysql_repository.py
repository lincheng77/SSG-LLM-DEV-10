from sqlalchemy import text
from sqlalchemy.ext.asyncio.session import AsyncSession


class DWMysqlRepository:
    """
    操作DW Mysql数据的仓库类
    """
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_column_types(self, table_name: str) -> dict[str, str]:
        """
        获取表的列类型
        """
        sql = f"show columns from {table_name}"
        # text() 函数将字符串转换为SQL文本，避免SQL注入攻击
        result = await self.session.execute(text(sql))
        return {row.Field: row.Type for row in result}

    async def get_column_values(self, table_name: str, column_name: str, limit: int = 10) -> list:
        """
        获取表的列值
        """
        sql = f"select distinct {column_name} from {table_name} where {column_name} is not null limit {limit}"
        result = await self.session.execute(text(sql))

        # scalars() 方法返回一个迭代器，可以使用 all() 方法将其转换为列表
        return result.scalars().all()

    async def get_db_infos(self) -> dict[str, str]:
        """
        获取数据库信息
        """
        result = await self.session.execute(text("select version()"))
        version = result.scalar() # 支持快捷获取单个值
        dialect = self.session.get_bind().dialect.name
        return {"version": version, "dialect": dialect}

    
    async def validate_sql(self, sql: str) -> str | None:
        """
        通过 EXPLAIN 校验SQL语法与表/字段引用是否合法（不真正执行）
        返回错误信息，通过则返回 None
        """
        try:
            await self.session.execute(text(f"EXPLAIN {sql}"))
            return None
        except Exception as e:
            await self.session.rollback()
            return str(e)

    async def execute_sql(self, sql: str):
        """
        执行SQL语句
        """
        result = await self.session.execute(text(sql))
        
        return [dict(row) for row in result.mappings().all()]


"""
.fetchall()  所有行的列表，每行是 Row（像元组，也能按列名取值）      [(1,'阿司匹林'), (2,'布洛芬'), (3,'感冒灵')]
.first()     只取第一行（Row），没查到返回 None                    (1,'阿司匹林')
.scalar()    第一行第一列的单个值，没查到返回 None                  1
.scalars()   每行的第一列组成的流，all() 收成列表                   [1,2,3]
.mappings()  每行转成字典的流，all() 收成字典列表                   [{'id':1,'name':'阿司匹林'}, {'id':2,'name':'布洛芬'}, ...]
"""
