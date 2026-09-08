from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_ import resolve_input_file


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:

    # 在进度板上把 node_entry 标记为"正在运行"(前端进度条显示为运行中)
    # TODO: is_stream=True 时可在节点变化时实时推送进度, 当前未启用
    add_running_task(state["task_id"], "node_entry")

    state = resolve_input_file(state)

    # 把 node_entry 从"正在运行"移到"已完成"(前端进度条显示为已完成)
    add_done_task(state["task_id"], "node_entry")

    return state