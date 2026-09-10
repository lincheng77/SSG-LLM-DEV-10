尚硅谷大模型第10期 第17部分 掌柜智库 章节

## 项目结构

```bash
app/
    ├─ api/                           # 接口层
    │  ├─ http/                       # 导入/查询两个可运行 server 入口
    │  └─ schemas/                    # 请求与响应数据结构
    ├─ infra/                         # 基础设施正式出口
    │  ├─ providers/                  # 配置聚合门面（infra_config，聚合 shared/config 各配置对象）
    │  ├─ document_parse/             # PDF 解析服务门面（MinerU）
    │  ├─ llm/                        # LLM / Embedding / Reranker 提供者出口
    │  ├─ object_storage/             # 对象存储门面（MinIO）
    │  ├─ persistence/                # 持久化门面（聊天历史等）
    │  └─ vectorstore/                # 向量库门面（Milvus）
    ├─ process/                       # 流程编排层
    │  ├─ import_/                    # 导入图、导入状态与导入页面
    │  │  ├─ agent/                   # LangGraph 导入图与各节点
    │  │  └─ page/                    # 导入演示页面
    │  └─ query/                      # 查询图、查询状态与查询页面
    │     ├─ agent/                   # LangGraph 查询图与各节点
    │     └─ page/                    # 查询演示页面
    ├─ rag/                           # RAG 核心能力层
    │  ├─ import_/                    # 导入域能力：入口识别、图片增强、切块、向量化、入库
    │  └─ query/                      # 查询域能力：主体确认、检索、HyDE、RRF、Rerank、答案生成
    ├─ resources/                     # 应用资源目录
    │  └─ prompts/                    # 提示词模板
    ├─ shared/                        # 公共底座
    │  ├─ clients/                    # 底层客户端工具（Milvus / Mongo / MinIO）
    │  ├─ config/                     # 原子配置读取与各组件配置对象
    │  ├─ model/                      # 模型工具封装
    │  ├─ runtime/                    # 运行时能力（日志、Prompt 加载）
    │  ├─ tool/                       # 模型下载等辅助脚本
    │  └─ utils/                      # 通用工具（SSE、任务状态、限流、路径等）
    └─ __init__.py
test/                                 # 测试与实验脚本
doc /                                 # 项目测试文档
output /                              # 项目文件生成存储位置
logs /                                # 项目日志文件位置
.env /                                # 项目的配置文件
.env.example /                        # 项目配置示例,传入到git仓库
```
