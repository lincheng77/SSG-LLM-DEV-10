
### 启动zero
nohup accelerate launch --config_file ./config/default_config.yaml ./src/train_zero.py 1>./logs/train_zero.log 2>&1 &