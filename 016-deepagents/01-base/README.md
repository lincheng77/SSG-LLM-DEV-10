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

## 后端 / 权限 / 技能篇

| 脚本名 | 主要学习什么 |
|---|---|
| 08_deep_agents_hitl_stream.py | 人机交互（HITL）之**流式审批**：用 `stream()` 边执行边输出中间过程，碰到 `interrupt_on` 拦下的高危工具时暂停，再用 `Command(resume=...)` 恢复 |
| 09_deep_agents_hitl_respond.py | 人机交互（HITL）之**直接回复（respond）**：审批时不执行工具，而是把人工提供的内容（模拟前端收集到的用户信息）当作工具结果直接回给模型 |
| 10_deep_agents_backend_filesystembackend.py | 后端之 `FilesystemBackend`：文件落到真实磁盘目录（`agent_workspace/`），`virtual_mode=True` 阻止路径穿越；注意文件是全局共享的，`thread_id` 只隔离会话状态、不隔离文件 |
| 11_deep_agents_backend_storebackend.py | 后端之 `StoreBackend`：文件存入 LangGraph `BaseStore`（`InMemoryStore`），由 `namespace` 工厂决定存放位置，实现**跨线程**持久化 |
| 12_deep_agents_backend_storebackend_dync_namespace.py | 后端之**动态命名空间**：`namespace` 工厂收到的是 `Runtime` 本身，按 `runtime.context.account` 动态算出 namespace，配合 `invoke(context=User(...))` 实现按用户隔离 |
| 13_deep_agents_backend_compositeBackend.py | 后端之 `CompositeBackend`：按路径前缀分流，`/store/` 走 `StoreBackend`、其余走默认的 `FilesystemBackend`，`ls` 会聚合成统一的文件视图 |
| 14_deep_agents_permissions_single.py | 权限之**单条规则**：`FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")` 禁止一切写入、读操作不受限；`deny` 只返回错误、**不会暂停图** |
| 15_deep_agents_permissions_multi.py | 权限之**多条规则组合**：先 `allow` 白名单目录（`/agent_workspace/**`）再 `deny` 全部（`/**`），规则按声明顺序**先命中先生效** |
| 16_deep_agents_skill_qiguasuanming.py | 技能（Skill）入门：用 `skills=["/skills/"]` 挂载技能目录，先让 agent 罗列出可用技能，再调用算命技能（`bazi-mingli`）完成任务 |
| 17_deep_agents_context.py | 技能 + **长期记忆**综合：挂载 `skills` 的同时用 `memory=["/project/AGENTS.md", "/project/preferences.md"]` 加载项目约定与用户偏好 |

> 公共约定：所有脚本从 `.env` 读取配置（DeepSeek 直连 + tavily key），中断类脚本（06～09）依赖 `checkpointer` + `thread_id` 存档恢复。
>
> 两类“拦下来”的机制不要混淆：`interrupt_on` / `mode="interrupt"` 会**暂停图**等人工审批（需 resume）；`mode="deny"` 只把错误**返回给模型**，图继续跑，模型可能反复重试。
