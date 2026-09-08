"""提示词加载：从 app/resources/prompts 读取 .prompt 文本并渲染占位符。"""
from functools import lru_cache

from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / 'app' / 'resources' / 'prompts'


@lru_cache(maxsize=None)
def _load_raw(name: str) -> str:
    """读取原始提示词文本（进程内缓存，避免重复磁盘 IO）。"""
    prompt_path = PROMPT_DIR / f'{name}.prompt'
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")
    return prompt_path.read_text(encoding='utf-8')


def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词；传入 kwargs 则渲染 {placeholder} 占位符
    :return: 渲染后的最终提示词字符串
    """
    raw = _load_raw(name)
    if not kwargs:
        return raw
    logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
    return raw.format(**kwargs)
