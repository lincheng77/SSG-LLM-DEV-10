import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from app.core.log import logger
from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState, DateInfoState, DBInfoState
from app.prompt.prompt_loader import load_prompt
from app.repositories.mysql.dw_mysql_repository import DWMysqlRepository


async def add_extra_context(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """补充生成SQL所需的上下文信息"""
    runtime.stream_writer({"stage": "添加额外信息"})

    dw_mysql_repository: DWMysqlRepository = runtime.context["dw_mysql_repository"]

    # 1. 收集当前日期、时间
    today = datetime.datetime.today()
    date = today.strftime("%Y-%m-%d")
    weekday = today.strftime("%A")
    quarter = f"Q{(today.month + 2) // 3}"

    date_info= DateInfoState(
        date = date,             # 年月日
        weekday = weekday,       # 星期
        quarter = quarter,       # 季度  Q1-Q4
    )

    # 2. 收集数据库信息
    db_info_mysql = await dw_mysql_repository.get_db_infos()
    db_info = DBInfoState(**db_info_mysql)

    logger.info(f"添加额外信息完成：date_info={date_info}, db_info={db_info}")

    return {"date_info_state": date_info, "db_info_state": db_info}

