from langchain_core.tools import tool
from ragflow_sdk import RAGFlow
from dotenv import load_dotenv
import os
from api.monitor import monitor

load_dotenv(override=True)


def load_rag_config() -> tuple[str]:
    base_url =os.getenv("RAGFLOW_API_URL")
    api_key = os.getenv("RAGFLOW_API_KEY")
    print(f"加载ragflow的配置: base_url:{base_url}  , api_key:{api_key}")
    return base_url,api_key


@tool
def search_assistant_list():
    """
    用于检索所有可用的助手信息!确定向哪个助手进行提问!
    返回结果: 没有: 没有检索到任何可使用的助手
             报错: 检索过程中发生异常,异常信息为:{str(e)}
             有  : 检索到助手信息: \n
                 名字: 名字 ,描述:描述 , 关联知识库: 1,2,3 \n
                 名字: 名字 ,描述:描述 , 关联知识库: 1,2,3 \n
                 名字: 名字 ,描述:描述 , 关联知识库: 1,2,3 \n
    :return:
    """
    monitor.report_tool(tool_name="search_assistant_list")
    try:
        # 1. 登录和连接ragflow服务器
        base_url,api_key = load_rag_config()
        ragflow = RAGFlow(base_url=base_url,api_key=api_key)

        # 2. 检索所有的助手信息
        # list[Chat]
        assistant_list = ragflow.list_chats()

        # 3. 循环助手列表,获取每个助手关联的知识库信息
        if not assistant_list:
            print(f"没有检索到任何可使用的助手")
            return "没有检索到任何可使用的助手"

        final_result = "检索到助手信息: \n"
        # 建立知识库 ID -> 名称 的映射 (当前 SDK 的 Chat 对象只有 dataset_ids 属性)
        dataset_name_map = {d.id: d.name for d in ragflow.list_datasets()}
        for assistant in assistant_list:
            name = assistant.name
            description = assistant.description
            dataset_ids = assistant.dataset_ids
            if dataset_ids:
                kb_names = ','.join([dataset_name_map.get(did, did) for did in dataset_ids])
            else:
                kb_names = '没有关联的知识库'
            final_result += (f"名字:{name},描述:{description},"
                             f"关联的知识库:{kb_names} \n")
        # 4. 拼接完整信息返回即可
        print(f"检索可用的助手:{final_result}")
        return final_result
    except Exception as e:
        # 5. 异常处理
        print(f"检索过程中发生异常,异常信息为:{str(e)}")
        return f"检索过程中发生异常,异常信息为:{str(e)}"



# 2. 向指定的助手进行提问
@tool
def ask_ragflow_assistant(assistant_name: str, query: str) -> str:
    """
      向指定助手进行提问,助手的名字通过: search_assistant_list进行确认!
      assistant_name助手的名字 query提问的问题
    :param assistant_name: 助手的名字
    :param query: 提问的问题
    :return: 直接返回提问的结果
    """
    monitor.report_tool(tool_name="ask_ragflow_assistant",args={"assistant_name":assistant_name,"query":query})
    try:
        # 1. ragflow的连接
        base_url,api_key = load_rag_config()
        ragflow = RAGFlow(base_url=base_url,api_key=api_key)

        # 2. 查找特定的助手
        chat_list =  ragflow.list_chats(name=assistant_name)
        if not chat_list:
            # 为空 -> 没有name对应的助手
            print(f"没有查找到:{assistant_name}对应第助手,请检查!")
            return f"没有查找到:{assistant_name}对应第助手,请检查!"
        chat  = chat_list[0]

        # 3. 创建会话
        chat_session = chat.create_session(name="temp")

        # 4. 会话提问
        """
           stream = True  for -> yield -> Message . content  for 
           stream = False     -> yield -> Message . content  next()
        
        """
        stream =  chat_session.ask(question=query,stream=False)
        # 5. 解析结果
        message = next(stream)
        content = message.content

        # 6. 关闭会话
        chat.delete_sessions(ids=[chat_session.id])

        # 7. 返回结果
        print(f"向:{assistant_name}提问,获取结果为:{content}")
        return content
    except Exception as e:
        # 5. 异常处理
        print(f"向:{assistant_name}提问的过程中发生异常,异常信息为:{str(e)}")
        return f"向:{assistant_name}提问的过程中发生异常,异常信息为:{str(e)}"

