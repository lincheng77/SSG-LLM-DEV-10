# 需求: 创建一个有个子代理的多智能体! 1. 负责算数 2. 负责翻译 3.负责查询天气(tavily)
"""
 1. 导入依赖
 2. 初始化模型
 3. 创建每个子代理
 4. 创建主代理
 5. 执行结果..
"""
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
import os

# 1. 加载配置文件
load_dotenv(override=True)


#2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

#3. 创建每个子代理
math_sub_agent = {
    "name":"math_sub_agent",
    "description":"用于处理算数问题的助手!所有算数问题都都可以调用我!!",
    "system_prompt":"我是一个算数专家,用于处理各种算数问题! 结果直接使用字符串形式输出皆可!!",
    "tools":[]
}

translate_sub_agent={
    "name":"translate_sub_agent",
    "description":"用于中英互相翻译的助手! 有翻译的问题找我!",
    "system_prompt":"我是翻译专家,可以将英文翻译成中文,可以将中文翻译成英文!!"
}


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def search_weather(city:str,max_results:int=1):
    """
     用于查询指定城市的天气信息! city 是查询天气的城市!!
    """
    print(f"本次调用工具,要查询城市的信息:{city}")
    return tavily_client.search(query=f"请帮我查询:{city}城市的天气信息!",max_results=max_results)


weather_sub_agent = {
    "name":"weather_sub_agent",
    "description":"用于查询指定城市的天气助手,查询天气需手求调用此助!",
    "system_prompt":"我是一个查询天气助手,可以调用 search_weather工具,查询指定城市的天气信息! 注意: 主要有查询天气需求,调用search_weather工具!",
    "tools":[search_weather]
}

# 4. 创建主智能体
main_agent = create_deep_agent(
    model=model,
    system_prompt="""
       我们是一个综合多智能体,可以处理天气,算数,以及翻译的任务!
       内置三个智能体(助手):  math_sub_agent 负责算数问题! translate_sub_agent负责中英翻译! weather_sub_agent负责查询指定城市天气!
       如果有相关的问题,必须调用子助手处理,主智能体不复杂具体的业务!!
    """,  # 模型
    subagents=[math_sub_agent,weather_sub_agent,translate_sub_agent],
    tools=[]
)


def execute_agent(query:str):
    stream =  main_agent.stream(input={
        "messages":[
            {
                "role":"user",
                "content":query
            }
        ]
    })

    for chunk in stream:
        # {"model/tools" : {messages:[]}}
        for node_name,state in chunk.items():
            if (not state) or ("messages" not in state): continue
            message =  state['messages'][-1]
            if node_name == "model":
                if message.content:
                    # 调用模型最终返回结果
                    print(f"[模型]调用模型最终返回结果:{message.content[:200]}...")
                else:
                    # 调用模型,决定调用工具或者子智能体
                    tool_calls = message.tool_calls
                    for tool_call in tool_calls:
                        tool_name = tool_call['name']
                        if tool_name == 'task':
                            # 调用子智能体
                            print(f"[模型]调用模型决定调用子智能体:{tool_call['args']['subagent_type']}子智能体,实现功能!")
                        else:
                            print(f"[模型]调用调用决定调用工具:{tool_name},参数:{tool_call['args']}实现功能!")
            elif node_name == "tools":
                name = message.name
                print(f"{name}*"*100)
                if name == 'task':
                    # 调用子智能体
                    print(f"[子智能体]调用子智能体,返回的结果:{message.content[:200]}...")
                else:
                    # 调用工具返回结果
                    print(f"[工具]调用工具:{name},返回的结果:{message.content[:200]}...")



if __name__ == '__main__':
    execute_agent("请帮我查询:上海的天气")
    execute_agent("请帮我将下面的中文翻译成英文: 黄焖鸡是世界上最好吃的食物?")
    execute_agent("请帮我进行算数运算: 998+889=?")
