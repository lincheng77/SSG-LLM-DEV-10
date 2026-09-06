# Docker 服务统一管理

业务 MySQL(pharma_db) + RAGFlow 全套，统一在本目录用 docker compose 管理。

compose 项目名在 docker-compose.yml 顶部用 `name: S10-016-deepagents-02-deep` 固定，避免与其他人目录名同为 docker 的项目在管理界面混在一起。

## 目录结构

```
docker/
├── docker-compose.yml            # 统一入口
├── .env                          # RAGFlow 配置(端口/密码/文档引擎)
└── ragflow/                      # RAGFlow compose 与配置文件
```

## 常用命令

```bash
cd 016-deepagents/02-deep/docker

docker compose up -d              # 启动全部(业务 mysql + RAGFlow, 共 6 个容器)
docker compose ps                 # 查看容器状态
docker compose logs -f ragflow    # 跟踪 RAGFlow 日志
docker compose logs -f mysql      # 跟踪业务 MySQL 日志
docker compose down               # 全部停止(数据保留在 named volume, 加 -v 连数据一起删)

docker compose up -d mysql                        # 只启动业务 MySQL
docker compose up -d es01 minio redis ragflow     # 只启动 RAGFlow 全套(ragflow-mysql 随依赖自启)
```

## 服务与端口

| 容器 | 镜像 | 宿主端口 | 用途 |
|---|---|---|---|
| deep-mysql | mysql:8.4 | 3306 | 业务库 pharma_db(drugs/inventory/sales_records) |
| ragflow-server | edwardelric233/ragflow:oc9 | 80 / 443 / 9380 | RAGFlow Web 界面与 API |
| ragflow-es-01 | elasticsearch:8.11.3 | 1200 | 文档引擎(默认) |
| ragflow-mysql | mysql:8.0.39 | 5455 | RAGFlow 元数据 |
| ragflow-minio | minio(RELEASE.2023-12-20) | 9000 / 9001 | 对象存储 / 控制台 |
| ragflow-redis | valkey:8 | 16379 | 缓存队列 |

## RAGFlow 初始化(首次启动后)

1. 浏览器打开 `http://localhost`，注册并登录；
2. 用 `../ragflow/知识库文件/` 里的资料创建知识库并解析；
3. 创建聊天助手，生成 API Key；
4. 把地址和 Key 填进项目根目录 `.env` 的 `RAGFLOW_API_URL` / `RAGFLOW_API_KEY`。

## 文档引擎切换

默认 elasticsearch。改 `ragflow/.env` 里的 `DOC_ENGINE=infinity` 后重新 `docker compose up -d`。
