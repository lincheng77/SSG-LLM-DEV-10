import tiktoken
import torch
from tiktoken_ext.openai_public import gpt2

from utils import get_device, generate
from model import GPT2


device = get_device()
# model_path = "./pretrained_model.pth"
model_path = "./sft_model.pth"
max_new_tokens = 200
temperature = 1.0


def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n### Response:\n"


tokenizer = tiktoken.get_encoding("gpt2")

# 加载方式和保存方式必须配套：
# 保存时是 torch.save(model.state_dict(), path)
# 加载时就要先构建相同结构的模型，再 load_state_dict
model = GPT2(
    vocab_size=50257,
    max_context_len=256,
    embed_dim=384,
    n_head=6,
    n_layer=6,
    ff_dim=4 * 384,
    dropout_rate=0.1,
)
model.load_state_dict(torch.load(model_path))
model.to(device)

while True:
    # 接受用户在中断的输入
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    # 格式化用户的提示词然后生成回答
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
    

    # 提取助手的回答部分
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 检查回答中是否包含换行符，切换输出格式
    if "\n" in response:
        print(f"Bot: \n{response}")
    else:
        print(f"Bot: {response}")