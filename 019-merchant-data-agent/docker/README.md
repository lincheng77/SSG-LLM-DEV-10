# Docker 服务统一管理

商户数据 Agent 的基础中间件全套：MySQL + Elasticsearch(IK 分词) + Kibana + Qdrant + Embedding 向量服务，统一在本目录用 docker compose 管理。

compose 项目名在 docker-compose.yaml 顶部用 `name: S10-019-merchant-data-agent` 固定，避免与其他人目录名同为 docker 的项目在管理界面混在一起。

## 目录结构

```
docker/
├── docker-compose.yaml           # 统一入口
├── mysql/                        # 业务库初始化脚本(首次启动自动执行)
│   ├── dw.sql
│   └── meta.sql
├── elasticsearch/                # ES 自定义镜像(内置 IK 分词器)
│   ├── Dockerfile
│   └── plugins/elasticsearch-analysis-ik-8.19.10.zip
└── embedding/                    # 本地向量模型文件
    └── bge-large-zh-v1.5/
```

## 常用命令

```bash
cd 019-merchant-data-agent/docker

docker compose up -d                # 启动全部(首次会自动构建 elasticsearch 镜像)
docker compose ps                   # 查看容器状态
docker compose logs -f mysql        # 跟踪 MySQL 日志
docker compose logs -f elasticsearch
docker compose down                 # 全部停止(数据保留在 named volume, 加 -v 连数据一起删)

docker compose up -d mysql          # 只启动 MySQL
docker compose up -d qdrant embedding   # 只启动向量相关服务
docker compose build elasticsearch  # 更换 IK 插件后重新构建 ES 镜像
```

## 服务与端口

| 容器 | 镜像 | 宿主端口 | 用途 |
|---|---|---|---|
| mysql | mysql:8.0 | 3306 | 业务库(root 密码 Atguigu.123, utf8mb4) |
| elasticsearch | docker-elasticsearch:latest(自建, ES 8.19.10 + IK) | 9200 | 全文检索/搜索服务 |
| kibana | kibana:8.19.10 | 5601 | ES 可视化管理界面 |
| qdrant | qdrant/qdrant:v1.16 | 6333 / 6334 | 向量数据库(HTTP / gRPC) |
| embedding | text-embeddings-inference:cpu-1.8 | 8081 | bge-large-zh-v1.5 文本向量服务 |

## 说明

- **MySQL**：`mysql/` 下的 SQL 脚本挂载到 `/docker-entrypoint-initdb.d`，只在数据卷首次初始化时自动执行；已初始化过再改脚本不会重新执行，需 `docker compose down -v` 后重启。
- **Elasticsearch**：单节点、禁用安全认证；IK 分词器版本必须与 ES 版本(8.19.10)一致，插件 zip 放在 `elasticsearch/plugins/` 下由 Dockerfile 安装。
- **Embedding**：模型从本地 `embedding/bge-large-zh-v1.5` 挂载加载，无需联网下载；验证：
  ```bash
  curl http://localhost:8081/embed -X POST \
       -H 'Content-Type: application/json' \
       -d '{"inputs":"测试文本"}'
  ```
