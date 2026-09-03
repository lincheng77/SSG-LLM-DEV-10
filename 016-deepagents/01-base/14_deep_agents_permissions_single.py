from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

import os

from deepagents import create_deep_agent,FilesystemPermission
from deepagents.backends import StateBackend,StoreBackend,FilesystemBackend,CompositeBackend
from pathlib import Path


# 1. 加载配置文件
load_dotenv(override=True)

# 2. 初始化模型(
model = init_chat_model(
    model=os.getenv("LLM_DEEPSEEK_PRO"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


root_dir_path:Path = Path(__file__).parent / "agent_workspace"
file_backend = FilesystemBackend(root_dir=root_dir_path,virtual_mode=True)

deep_agent = create_deep_agent(
    model = model,
    system_prompt="""
      你是一个综合助手!

      # 当前权限约束(必读)
      文件系统权限已配置: write 操作对所有路径(/**)一律 deny。
      也就是说任何写入/编辑都会被拒绝, 工具会返回:
      "Error: permission denied for write on ..."。
      读操作(read/ls/glob/grep)不受限制, 可以正常使用。

      # 碰到权限拒绝时的行为准则(最重要)
      1. 收到 permission denied 后绝对不要重试: 不要重写一次, 不要换文件名或路径再试,
         也不要改用 edit_file 等其他写入工具绕过。
      2. 立即终止当前任务, 直接告知用户: 该操作被权限规则拒绝, 无法完成。
      3. 需要写入才能完成的任务, 直接返回失败说明即可, 不要尝试任何变通方案，没有任何变通方案，告知被拒绝了就行，不用做其他任何尝试。

      # 能力边界
      你没有联网搜索工具, 无法查询外部资料。遇到需要外部信息的请求,
      直接说明自己没这个能力, 不要凭空编造内容后去写文件。
    """,
    backend=file_backend,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny"
        )
    ]
)

result1 = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="读取info.txt文件,大风哥今年多大了!")
        ]
    }
)
print(f"读取的结果:{result1['messages'][-1].content}")

result2 = deep_agent.invoke(
    input = {
        "messages":[
            HumanMessage(content="帮我查询python语言的发展历史,写入到python.md文件中!")
        ]
    }
)
print(f"写入的结果:{result2['messages'][-1].content}")
