import os
from dotenv import load_dotenv
from api.monitor import monitor
from mysql.connector import connect, Error
from langchain_core.tools import tool

load_dotenv()


def get_db_config():
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        # 排序规则: ci = 大小写不敏感
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        # 自动提交, 无需 commit()
        "autocommit": True,
        # 严格模式: 非法数据直接报错而不是静默截断
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL")
    }
    # 移除 None 值（核心必要操作）
    config = {k: v for k, v in config.items() if v is not None}

    # 补充：校验核心配置是否存在（可选但推荐）
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


@tool
def select_tables():
    """
      查询数据库中表信息! 用于后续的自定义sql语句提供存在的表名支撑!
        返回结果的格式:
           1. 存在   return "可用的表名为: 1,2,3,4"
           2. 不存在 return "当前库没有任何可用的表"
           3. 报错  return "查询表名过程中报错了,错误信息为:{x}"
    :return:
    """

    monitor.report_tool(tool_name="select_tables",args={})

    try:
        # 1.获取连接
        with connect(**get_db_config())  as conn:
            with conn.cursor() as cursor:
                sql = "show tables;"
                cursor.execute(sql)
                result = cursor.fetchall()

                if not result:
                    # 空 没有查询到表信息
                    print(f"当前库没有任何可用的表")
                    return "当前库没有任何可用的表"
                table_names = [item[0] for item in result]
                print(f"可用的表名为: {",".join(table_names)}")
                return f"可用的表名为: {",".join(table_names)}"
    except Error as e:
        print(f"查询表名过程中报错了,错误信息为:{str(e)}")
        return f"查询表名过程中报错了,错误信息为:{str(e)}"



@tool
def select_table_data(table_name:str):
    """
      查询指定表名的至多前100条数据! 通过查询数据确定表字段和表关系!
      表设计的时候,主外键的命名相同,非主外键命名不同!
      table_name要查询的表名!
      有数据: return CSV格式  列名,列名,列名\n 值1,值2,值3 \n 值1,值2,值3 \n
      没有数据: return 查询{table_name}没有数据!
      报错:  return 查询{table_name}的过程报错,错误信息为:{str(e)}

    :param table_name: 表名
    :return: 返回的结果格式化成CSV(表格)格式
    """

    monitor.report_tool(tool_name="select_table_data",args={"table_name":table_name})

    try:
        # 1.创建连接
        with connect(**get_db_config()) as conn:
            with conn.cursor() as cursor:
                sql = f"select * from {table_name} limit 100;"
                cursor.execute(sql)

                result = cursor.fetchall()
                description =  cursor.description #所有列属性
                if not result:
                    print(f"查询{table_name}没有数据")
                    return f"查询{table_name}没有数据"

                csv_head_str =  ",".join([t[0] for t in description]) #所有列名

                csv_body_str_lines = [",".join(map(str,t)) for t in result]
                csv_body_str = "\n".join(csv_body_str_lines)
                return f"{csv_head_str}\n{csv_body_str}"
    except Error as e:
        print(f"查询{table_name}过程中报错了,错误信息为:{str(e)}")
        return f"查询{table_name}过程中报错了,错误信息为:{str(e)}"


@tool
def execute_custom_select_sql(sql: str):
    """
    :param sql: 自定义sql
    :return: 返回的结果格式化成CSV(表格)格式
    """

    monitor.report_tool(tool_name="execute_custom_select_sql", args={"sql": sql})

    try:
        # 1.创建连接
        with connect(**get_db_config()) as conn:
            with conn.cursor() as cursor:

                cursor.execute(sql)
                result = cursor.fetchall()
                description = cursor.description

                if not result:
                    print(f"执行:{sql}没有查询到数据")
                    return f"执行:{sql}没有查询到数据"

                csv_head_str: str = ",".join([t[0] for t in description])

                # [(1, '阿司匹林', 12.5)] -> ['1,阿司匹林,12.5']
                csv_body_str_lines = [",".join(map(str, t)) for t in result]

                # '1,阿司匹林,12.5'
                # '2,XXXXXX,1.39'
                csv_body_str = "\n".join(csv_body_str_lines)
                return f"{csv_head_str}\n{csv_body_str}"
    except Error as e:
        print(f"执行{sql}过程中报错了,错误信息为:{str(e)}")
        return f"执行{sql}过程中报错了,错误信息为:{str(e)}"
