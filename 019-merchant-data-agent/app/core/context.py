import asyncio
from _contextvars import ContextVar, Token

# 请求上下文变量
_req_context_var = ContextVar("req_id", default="")

def set_req_id(req_id: str) -> Token:
    return _req_context_var.set(req_id)

def get_req_id() -> str:
    return _req_context_var.get()

def reset_req_id(token: Token):
    _req_context_var.reset("")