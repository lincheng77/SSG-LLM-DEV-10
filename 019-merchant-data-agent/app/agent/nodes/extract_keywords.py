from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from jieba.analyse import extract_tags

from app.core.log import logger


async def extract_keywords(state: DataAgentState, runtime: Runtime[DataAgentContext]) -> dict:
    """从用户问题中抽取用于检索的关键词"""

    runtime.stream_writer({"stage": "提取关键字"})

    query = state["query"]

    # 使用jieba进行分词
    allow_pos = (
        "n",  # 名词: 数据、服务器、表格
        "nr",  # 人名: 张三、李四
        "ns",  # 地名: 北京、上海
        "nt",  # 机构团体名: 政府、学校、某公司
        "nz",  # 其他专有名词: Unicode、哈希算法、诺贝尔奖
        "v",  # 动词: 运行、开发
        "vn",  # 名动词: 工作、研究
        "a",  # 形容词: 美丽、快速
        "an",  # 名形词: 难度、合法性、复杂度
        "eng",  # 英文
        "i",  # 成语
        "l",  # 常用固定短语
    )

    result = extract_tags(query, topK=10, allowPOS = allow_pos)
    logger.info(f"提取的关键字为：{result}")

    # 用set是为了去重！！！！！！！
    keyswords = list(set(result + [query]))

    # 下面的代码等价
    # keyswords = list(set(result))
    # keyswords.append(query)


    logger.info(f"抽取关键字完成：{keyswords}")

    return {"keywords": keyswords}
