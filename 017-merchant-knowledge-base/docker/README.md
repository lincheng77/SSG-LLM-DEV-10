# docker 中间件环境

## 服务清单

| 容器名 | 镜像 | 端口 | 用途 |
|---|---|---|---|
| minio | quay.io/minio/minio | 9000 / 9001 | 图片对象存储(node_md_img 节点用) |

## 操作

```bash
cd 017-merchant-knowledge-base/docker
docker compose up -d          # 启动
docker compose logs -f minio  # 看日志
docker compose down           # 停止(数据保留在 named volume)
```

## MinIO 访问

- 对象存储地址: http://服务器IP:9000
- 控制台地址: http://服务器IP:9001
- 用户名/密码: minioadmin / minioadmin

## 项目根 .env 需要保持一致

```
MINIO_ENDPOINT=服务器IP:9000     # 不带 http:// ；代码用它拼接图片在线URL, 本机自用可填 127.0.0.1
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=enterprise-rag
MINIO_IMG_DIR=/kb-images
MINIO_SECURE=False
```

注意: 016-deepagents 的 RAGFlow 全家桶启动时也占用 9000/9001, 两边不要同时启动。
