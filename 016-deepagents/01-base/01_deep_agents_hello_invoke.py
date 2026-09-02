# 演示deepagents的深度代理智能体的创建!
# 演示tavily工具的使用

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
import os
from tavily import TavilyClient
from deepagents import create_deep_agent


#1. 加载配置
load_dotenv(override=True)


#2. 初始化模型(DeepSeek 官方直连, 配置在 .env 里)
# DeepSeek API 兼容 OpenAI 格式, 所以 provider 用 openai, 只需把 base_url/api_key 指向 DeepSeek
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 3.定义工具
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# tool -> agent -> 智能体(大模型)
@tool
def network_search(query:str,max_results:int=10):
    """
    进行联网搜索的工具, query参数要搜索的内容,max_results要搜索的条数!默认10条!
    :param query:  要搜索的内容
    :param max_results: 要搜索的条数
    :return:
    """
    # 联网搜索能力
    print(f"已经启动联网查询,查询问题:{query},查询的条数:{max_results}")
    # 联网搜索
    return  tavily_client.search(query=query,max_results=max_results)


# 4. 创建一个深度代理的智能体(本次只有工具没有子代理)
deep_agent = create_deep_agent(
    model= model,
    tools=[network_search],
    system_prompt="""
      你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 network_search 工具来收集信息。
    """
)

# 5. 执行深度代理智能体
# langchain -> invoke ainvoke / stream astream
# deepagents -> invoke ainvoke / stream astream
result = deep_agent.invoke(input={
    "messages":[
        {
            "role":"user",
            "content":"黄焖鸡是不是世界上最好吃的食物?黄焖鸡的发明人是谁?"
        }
    ]
})

print(result)

print(result["messages"][-1].content)