from pathlib import Path


# 加载提示词文件文本
def load_prompt(name: str):
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / f"{name}.prompt"

    return prompt_path.read_text("utf-8")
