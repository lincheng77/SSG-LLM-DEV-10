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
stream = deep_agent.stream(input={
    "messages":[
        #HumanMessage(content="xx")
        {
            "role":"user",
            "content":"黄焖鸡是不是世界上最好吃的食物?黄焖鸡的发明人是谁?"
        }
    ]
})

for chunk in stream:

    print(chunk)
    """
      场景1: 无效场景 {'PatchToolCallsMiddleware.before_agent': None} {'TodoListMiddleware.after_model': None} 
             {key确定:None / 没有messages}
      场景2: 调用模型模型决定调用哪些工具/subagent AIMessage content没有值,tool_calls=[{}] 
             {model: {messages:[AIMessage(content=没值,tool_calls=[有值])]}}
      场景3: ToolMessage 调用了工具/subagent 返回的结果
             {tools: {messages:[ToolMessage(content=有值,name="工具名")]}}
      场景4: AIMessage content有值,tool_calls没有值 调用模型,返回最终的结果
             {model: {messages:[AIMessage(content=有值,tool_calls=[没有值])]}}
    """
    for node_name,state in chunk.items():
        # 去掉无效的数据
        if (not state) or ("messages" not in state): continue
        message =  state['messages'][-1]
        # model返回信息
        if node_name == "model":
            if message.content:
                print(f"[模型]最终返回的结果:{message.content[:100]}....")
            else:
                # 模型决定调用工具或者调用子智能体
                tool_calls = message.tool_calls
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    if tool_name == 'task':
                        # 调用了子智能体
                        print(f"[模型]决定本次调用{tool_call['args']['subagent_type']}子智能体,实现功能! ")
                    else:
                        # 调用具体的工具
                        print(f"[模型]决定本次调用{tool_name}工具,实现功能!传入参数为:{tool_call['args']}")
        # 工具返回信息
        elif node_name == "tools":
            # tool
            name = message.name
            if name == 'task':
                # 调用子智能体
                print(f"[子智能体]执行返回的结果{message.content[:100]}...!")
            else:
                # 调用具体的工具
                print(f"[工具]{name}执行返回的结果:{message.content[:100]}...!")
