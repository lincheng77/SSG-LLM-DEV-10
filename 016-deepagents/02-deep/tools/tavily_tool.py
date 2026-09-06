from langchain_core.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from api.monitor import monitor

load_dotenv(override=True)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def internet_search(query:str,max_results:int=10):
    """
      进行联网搜索信息的工具! 专门用于查询数据库和rag知识库之外的数据!
    :param query: 查询的问题
    :param max_results: 返回的结果的数量,默认是10条
    :return: 查询的结果
    """
    print(f"[联网搜索智能体]internet_search工具开始被调用,参数为:{query},{max_results}")
    # 向前端发送消息!
    monitor.report_tool(tool_name="internet_search",args={"query":query,"max_results":max_results})
    result = tavily.search(
        query=query,
        max_results=max_results,
        include_answer=True  # 总结
    )
    print(f"[联网搜索智能体]internet_search工具调用结束,查询结果为:{result}")
    return result


