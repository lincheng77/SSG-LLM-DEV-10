from contextvars import ContextVar, Token
from typing import Optional

"""
ContextVar 会话上下文变量，跟着携程、线程一直走，不会丢，即使跨线程调用
"""

_session_dir_ctx: ContextVar[Optional[str]] = ContextVar("session_dir", default=None)

_session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def set_session_dir_context(session_dir: str) -> Token:
    return _session_dir_ctx.set(session_dir)


def get_session_dir_context() -> Optional[str]:
    return _session_dir_ctx.get()


def set_session_id_context(session_id: str) -> Token:
    return _session_id_ctx.set(session_id)


def get_session_id_context() -> Optional[str]:
    return _session_id_ctx.get()


def reset_session_context(session_dir_context_token: Token, session_id_context_token: Token = None):
    _session_dir_ctx.reset(session_dir_context_token)
    if session_id_context_token:
        _session_id_ctx.reset(session_id_context_token)
