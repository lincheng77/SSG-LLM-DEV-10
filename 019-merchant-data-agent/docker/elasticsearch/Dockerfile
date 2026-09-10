# 基于 Elasticsearch 8.19.10 官方镜像构建
FROM elasticsearch:8.19.10

# 切换到 root 用户以执行特权操作（安装插件需要 root 权限）
USER root

# 将宿主机上的 IK 分词器插件包复制到容器 /tmp 目录
COPY plugins/elasticsearch-analysis-ik-8.19.10.zip /tmp/

# 通过 ES 自带的插件管理工具安装 IK 分词器
# --batch 表示自动确认许可协议等交互提示，适合 Docker 构建的非交互环境
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch \
    file:///tmp/elasticsearch-analysis-ik-8.19.10.zip

# 将插件目录的归属改为 elasticsearch 用户，确保 ES 启动后能正常加载插件
RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/plugins

# 切回 elasticsearch 用户，以非 root 身份运行服务（官方镜像默认用户，安全最佳实践）
USER elasticsearch
