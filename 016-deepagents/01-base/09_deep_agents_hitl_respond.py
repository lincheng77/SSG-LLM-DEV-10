# 收集用户信息
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver  # 内存检查点，用于保存中断状态
from langgraph.types import Command  # 恢复执行的指令类型
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import InMemorySaver



# 1. 加载配置文件
load_dotenv(override=True)

#2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


@tool
def get_user_info():
    """
    用于收集用户信息的工具!!
    :return:
    """
    print("收集用户信息工具:get_user_info")
    return "哈哈哈哈"


checkpointer = InMemorySaver()
config={"configurable":{"thread_id":"007"}}
main_agent = create_deep_agent(
    model=model,
    tools=[get_user_info],
    system_prompt="""
      我们是一个综合助手! 内部: get_user_info工具,用于收集用户信息!
      如果有收集信息需求,必须调用get_user_info工具!
    """,
    checkpointer=checkpointer,
    interrupt_on={
        "get_user_info":True
    }
)



# 第一遍执行 查询用户信息...
result = main_agent.invoke(
    input={
        "messages":[
            HumanMessage(content="请帮我查询用户信息,说出用户的名字和年龄!!")
        ]
    },
    config=config)

print("第一遍执行......................................")
interrupts = result.get("__interrupt__")
if interrupts:
    print("============中间的放行逻辑============")
    action_requests = interrupts[0].value['action_requests']
    decisions = []
    for action_request in action_requests:
        # {name:工具名,args:{参数...}}
        name = action_request['name']
        if name == 'get_user_info':
            # 收集用户信息!
            # websocket / sse 向前端推送信息 -> 弹出一个框 -> 输入信息 -> 获取信息
            print("启动前端连接调用")
            print("返回了用户信息")
            user_info = "用户姓名为:赵伟风,年龄为:23"
            decisions.append(
                {
                    # respond:不执行工具,把 message 里的内容当作工具结果直接返回给模型
                    # 适用场景:审批时就能拿到答案(如这里模拟前端返回的用户信息),无需真调工具
                    # 其实应该把 user_info 当作工具结果返回给模型，这里只是为了学习respond
                    "type":"respond",
                    "message":user_info  # 本次动作的描述信息!!
                }
            )
        else:
            decisions.append(
                {
                    "type":"approve"
                }
            )

    # 第二次执行
    print("第二遍执行......................................")
    result = main_agent.invoke(
        Command(resume={"decisions":decisions}),config=config)
    print(f"最终执行结果:{result['messages'][-1].content}")
