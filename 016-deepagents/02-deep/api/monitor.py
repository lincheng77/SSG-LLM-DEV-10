import datetime
import asyncio
from typing import Any, Dict, Optional
from fastapi import WebSocket
from api.context import get_session_id_context

# 尝试导入全局运行时（用于脚本模式下的流式输出）
try:
    import builtins
except ImportError:
    builtins = None


class ToolMonitor:
    """
    工具监控类，用于在工具执行过程中上报进度和状态。
    设计为【单例模式】，可在任何工具中直接导入使用。
    兼容 FastAPI WebSocket 和 脚本运行时的 stream_writer。

    使用示例:
    from api.monitor import monitor
    """
    _instance = None

    # 这样做的目的是单例模式
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.websocket_manager = None  # 预留给 FastAPI WebSocketManager
        return cls._instance

    def set_websocket_manager(self, manager):
        """设置 FastAPI 的 WebSocket 管理器"""
        self.websocket_manager = manager

    def _emit(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """内部发送方法"""
        payload = {
            "type": "monitor_event",
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 以下三个输出渠道依次全部执行, 不是失败降级的 fallback 关系

        # 1. FastAPI WebSocket: 定向推送到当前会话
        self._push_to_websocket(payload)

        # 这个项目现在用不到
        # # 2. DeepAgents 脚本模式: 全局 runtime 的 stream_writer
        # #    (simple_agents.py 中的 MockRuntime 借此接收数据)
        # if builtins and hasattr(builtins, 'runtime') and hasattr(builtins.runtime, 'stream_writer'):
        #     try:
        #         builtins.runtime.stream_writer(payload)
        #     except Exception:
        #         pass

        # 3. 控制台保底输出: 带特殊前缀, 方便肉眼识别
        print(f"\n[Monitor:{event_type}] {message}")

    def _push_to_websocket(self, payload: Dict[str, Any]) -> None:
        """通过 FastAPI WebSocket 将 payload 定向推送到当前会话。
        """
        manager = self.websocket_manager
        manager_loop = getattr(manager, "loop", None) if manager else None
        if not manager_loop:
            return

        session_id = get_session_id_context()
        if not session_id:
            return


        # 跨线程投递: 工具都是同步函数, 经 astream 会被扔进线程池执行,
        # 此处可能在线程池线程上, 必须用 run_coroutine_threadsafe 线程安全地送回 loop
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_to_session(payload, session_id),
                manager_loop  # uvicorn 的事件循环
            )
        except Exception as e:
            print(f"[Monitor] WebSocket send failed: {e}")

    def report_tool(self, tool_name: str, args: Dict[str, Any] = None):
        """报告工具开始执行"""
        self._emit("tool_start", f"开始执行工具: {tool_name}", {"tool_name": tool_name, "args": args})

    def report_assistant(self, assistant_name: str, args: Dict[str, Any] = None):
        """报告正在调用的子智能体进度"""
        self._emit("assistant_call", f"正在调用助手: {assistant_name}",
                   {"assistant_name": assistant_name, "args": args})

    def report_task_result(self, result: str):
        """报告任务最终结果"""
        self._emit("task_result", "任务执行完成", {"result": result})

    def report_session_dir(self, path: str):
        """报告任务工作目录"""
        self._emit("session_created", f"工作目录已创建: {path}", {"path": path})


# 全局单例实例
monitor = ToolMonitor()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # 延迟绑定 loop，防止初始化时 loop 不一致
        self.loop = None

    def set_loop(self, loop):
        """显式设置事件循环"""
        self.loop = loop
        monitor.set_websocket_manager(self)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"Client connected: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        print(f"Client disconnected: {session_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_session(self, message: dict, session_id: str):
        """向指定会话推送消息; 连接已失效时自动清理。

        本方法通常由 `run_coroutine_threadsafe` 投递执行, 返回的 Future 没人接收,
        所以异常必须在内部消化, 否则 asyncio 会刷
        "Future exception was never retrieved" 警告。
        """
        websocket = self.active_connections.get(session_id)
        if websocket is None:
            # 会话未连接或已断开, 静默丢弃
            return

        try:
            await websocket.send_json(message)
        except Exception as e:
            # 写入失败说明连接已死: 清掉映射, 避免后续持续往断开的连接上写
            self.active_connections.pop(session_id, None)
            print(f"[WebSocket] send to {session_id} failed, connection dropped: {e}")


manager = ConnectionManager()
