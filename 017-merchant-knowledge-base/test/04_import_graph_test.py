# 动态测试
from app.shared.runtime.logger import logger
from app.process.import_.agent.state import create_default_state
from app.process.import_.agent.main_graph import bk_import_app
import json

state = create_default_state(task_id="123456", input_file_path="test.pdf")
logger.info(f"开始进行测试,输入的数据:{json.dumps(state, indent=4, ensure_ascii=False)}")
result = bk_import_app.invoke(state)
logger.info(f"测试结束,输出的数据:{json.dumps(result, indent=4, ensure_ascii=False)}")
