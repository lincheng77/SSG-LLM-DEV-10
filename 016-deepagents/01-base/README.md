# 01-base DeepAgents 基础入门

| 脚本名 | 主要学习什么 |
|---|---|
| 01_deep_agents_hello_invoke.py | 创建第一个深度代理智能体（deep agent）：定义 tavily 联网搜索工具，用 `invoke` 同步执行拿最终结果 |
| 02_deep_agents_hello_stream.py | 与 01 相同的智能体，改用 `stream` 流式执行，实时输出中间过程 |
| 03_deep_agents_subagents_dict.py | 多智能体入门：用**字典**方式定义多个子代理（算数/翻译/天气），由主代理自动调度 |
| 04_deep_agents_subagents_compiledsubagent_langchain.py | 用 `CompiledSubAgent` 把一个 **langchain** 的 `create_agent` 包装成子代理 |
| 05_deep_agents_subagents_compiledsubagent_langgraph.py | 用 `CompiledSubAgent` 把一个 **langgraph** 编译的 StateGraph 图包装成子代理（含 START/END、入口节点等图基础） |
| 06_deep_agents_hitl_approve_reject.py | 人机交互（HITL）之**审批/拒绝**：`interrupt_on` 拦截高危工具，`approve` 放行、`reject` 拒绝，`Command(resume=...)` 恢复执行 |
| 07_deep_agents_hitl_edit.py | 人机交互（HITL）之**编辑参数**：审批时用 `edit` 修改工具入参再放行，模型对人工干预无感知 |

> 公共约定：所有脚本从 `.env` 读取配置（DeepSeek 直连 + tavily key），中断类脚本（06/07）依赖 `checkpointer` + `thread_id` 存档恢复。
