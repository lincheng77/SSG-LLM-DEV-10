import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle

from model import GPT2
from utils import get_device
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import tiktoken

# 设备
device = get_device()
# 预训练模型保存的路径
pretrained_model_path = "./pretrained_model.pth"

# 设置超参数
context_len = 256
vocab_size = 50257
batch_size = 32
learning_rate = 3e-4
max_iters = 2000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1


class TokenDataset(Dataset):
    def __init__(self, text, context_len):

        tokenizer = tiktoken.get_encoding("gpt2")  # gpt2 的分词器

        # 分词
        tokens = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        # 输入
        x = self.tokens[idx : idx + self.context_len]
        # 预测目标
        y = self.tokens[idx + 1 : idx + self.context_len + 1]
        return x, y


# 将训练数据读入内存
with open("tiny_codes.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = TokenDataset(text, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 初始化模型和优化器
model = GPT2(
    vocab_size = vocab_size,
    max_context_len = context_len,
    embed_dim = embed_dim,
    n_head = n_head,
    n_layer = n_layer,
    ff_dim = ff_dim,
    dropout_rate = dropout_rate,
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# 参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"参数数量：{total_params}，（{total_params/1e6:.1f}M）")


losses = []
data_iter = cycle(dataloader) # 无线循环的使用数据集
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # logits
    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
         batch_y.view(-1))

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
plt.savefig("loss_pretrain.png")


torch.save(model.state_dict(), pretrained_model_path)
