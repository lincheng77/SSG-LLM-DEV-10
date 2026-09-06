
from yaml import safe_load
from pathlib import Path


config_path_obj = Path(__file__).resolve().parents[1] / "prompt" / "config.yaml"


config_dict = {}
with open(config_path_obj, "r", encoding="utf-8") as stream:
    config_dict = safe_load(stream)





