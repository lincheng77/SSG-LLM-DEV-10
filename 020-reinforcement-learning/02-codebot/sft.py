import json
from itertools import cycle

import tiktoken
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
# 配置
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GPT2
from torch.optim import AdamW
from utils import get_device


device = get_device()
data_path = "./tiny_codes_sft.json"
pretrain_model_path = "./pretrained_model.pth"
sft_model_save_path = "./sft_model.pth"

# 超参数
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path, "r") as f:
            data = json.load(f)
            
        for item in data:
            ids, labels = self._create_sample(
                item["instruction"],
                item["response"]
            )
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dis, labels = self.samples[idx]
        return torch.tensor(dis, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        
    def _create_sample(self, instruction, response):
        
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"
        response = f"{response}<|endoftext|>"

        # 分词（tiktoken 的参数是 allowed_special，不是 add_special_tokens）
        prompt_ids = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        response_ids = self.tokenizer.encode(response, allowed_special={"<|endoftext|>"})

        # 创建输入序列和标签
        # 标签中的提示词部分使用-100 进行掩码
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 进行错位！
        ids= ids[:-1]
        labels= labels[1:]

        # 根据context_len 进行填充或截断
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            # 填充
            ids = ids + [0] * pad_len               # 填充0表示padding
            labels = labels + [-100] * pad_len      # 填充符不计算损失
        elif pad_len < 0:
            # 截断
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]
        

        return ids, labels


# 分词器和数据集的准备
tokenizer = tiktoken.get_encoding("gpt2")
dataset = SFTDataset(data_path, tokenizer, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        

# 模型和优化器
model = GPT2(
    vocab_size=50257,
    max_context_len=context_len,
    embed_dim=384,
    n_head=6,
    n_layer=6,
    ff_dim=4 * 384,
    dropout_rate=0.1,
)
model.to(device)
# SFT 的起点是预训练权重，而不是随机初始化
model.load_state_dict(torch.load(pretrain_model_path))
optimizer = AdamW(model.parameters(), lr=learning_rate)


# 训练
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device) # shape: (batch_size, context_len)
    
    logits = model(batch_x)   # shape: (batch_size, context_len, vocab_size)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), # shape: (batch_size * context_len, vocab_size)
        batch_y.view(-1),   # shape: (batch_size * context_len,)
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    pbar.set_postfix({"loss" : f"{loss.item():.4f}"})

# 保存图像
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_sft.png")

torch.save(model.state_dict(), sft_model_save_path)