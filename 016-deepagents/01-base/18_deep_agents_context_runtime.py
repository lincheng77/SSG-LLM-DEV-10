
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
import os
from deepagents import create_deep_agent,FilesystemPermission

from dataclasses import dataclass
from langgraph.prebuilt import ToolRuntime


# 1. 加载配置文件
load_dotenv(override=True)

# 2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


@dataclass
class User:
    account:str # 登录系统的账号
    name:str # 登录系统的名字

@tool
def transfer(account_id: str, amount: int , runtime:ToolRuntime[User]) -> str:

    """
    实现转账业务的工具! 先account_id账号转账! amount转账金额!
    """

    account =  runtime.context.account
    name    =  runtime.context.name

    if account in ("admin","root"):
        print(f"当前登录账号:{account},姓名:{name},是管理员,可以发起转账!!")
        print(f"转账成功,已经向:{account_id}账号,转入:{amount}元")
        return (f"转账成功,已经向:{account_id}账号,转入:{amount}元")
    else:
        print(f"当前登录账号:{account},姓名:{name},没有权限转账!!")
        return (f"当前登录账号:{account},姓名:{name},没有权限转账,向{account_id}转账失败!!")

deep_agent = create_deep_agent(
    model = model,
    system_prompt="""
      我们是一个银行转账助手! 可以使用transfer工具进行转账!
      转账工具中有权限判定,可能出现转账失败的情况!
    """,
    tools=[transfer]
)


result1 = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "我们要向: zhaoweifeng 账号,转账:270块钱"}]},
    config={"configurable": {"thread_id": "demo_1"}},
    context=User(account="admin",name="二狗哥") # 每次执行的上下文
)
print(f"[Agent]: {result1['messages'][-1].content}")



result2 = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "我们要向: mayun 账号,转账:1000块钱"}]},
    config={"configurable": {"thread_id": "demo_1"}},
    context=User(account="test",name="蛋蛋哥") # 每次执行的上下文
)
print(f"[Agent]: {result2['messages'][-1].content}")


