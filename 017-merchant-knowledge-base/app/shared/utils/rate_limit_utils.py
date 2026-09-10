from langchain_core.rate_limiters import InMemoryRateLimiter

from app.shared.runtime.logger import logger  # 复用项目全局logger

# 按参数组合缓存限流器实例（令牌桶需长期持有才能正确积累令牌）
_LIMITER_CACHE: dict[tuple[int, int], InMemoryRateLimiter] = {}


def _get_limiter(max_requests: int, window_seconds: int) -> InMemoryRateLimiter:
    """获取或创建对应参数组合的令牌桶限流器"""
    key = (max_requests, window_seconds)
    if key not in _LIMITER_CACHE:
        _LIMITER_CACHE[key] = InMemoryRateLimiter(
            requests_per_second=max_requests / window_seconds,
            max_bucket_size=max_requests,
        )
    return _LIMITER_CACHE[key]


def apply_api_rate_limit(
        max_requests: int = 500,
        window_seconds: int = 60
) -> None:
    """
    通用API速率限制器（基于langchain_core的InMemoryRateLimiter令牌桶实现，线程安全）
    """
    limiter = _get_limiter(max_requests, window_seconds)
    # 桶空时同步阻塞直到有令牌可用
    limiter.acquire()
    logger.debug(f"API请求已通过限流器，配额：{window_seconds}秒内最多{max_requests}次")
