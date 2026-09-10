"""
工具模块，负责提供 lm 相关的辅助能力。
"""
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel

from app.shared.config.llm_config import llm_config
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen3-32b"
_DEFAULT_LLM_TEMPERATURE = 0.1
_DEFAULT_LLM_PROVIDER = "openai"
_llm_client_cache: dict[tuple[str, bool], BaseChatModel] = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> BaseChatModel:
    """
    param model: 模型名称，优先级：传入参数 > 配置文件 llm_config.llm_model > 内置默认模型
    param json_mode: 是否开启JSON输出模式，开启后返回标准json_object格式（适配结构化数据解析）
    return: 初始化完成的聊天模型实例（优先从全局缓存获取，未命中则新建并缓存）
    """
    # 1. 确定目标模型（优先级递减，保证模型名非空）
    target_model = model or llm_config.llm_model or _DEFAULT_LLM_MODEL


    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    cache_key = (target_model, json_mode)
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not llm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not llm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_BASE_URL（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    extra_body = {"enable_thinking": False}  # 千问专属：关闭思考链输出，减少冗余内容
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 5. 客户端初始化：init_chat_model 统一工厂入口，按 provider 选类（此处固定 openai 兼容协议）
    try:
        llm_client = init_chat_model(
            model=target_model,  # 目标模型名
            model_provider=llm_config.llm_provider or _DEFAULT_LLM_PROVIDER,
            temperature=llm_config.llm_temperature or _DEFAULT_LLM_TEMPERATURE,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            extra_body=extra_body,  # 国产模型私有参数透传
            model_kwargs=model_kwargs,  # OpenAI通用参数
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client

