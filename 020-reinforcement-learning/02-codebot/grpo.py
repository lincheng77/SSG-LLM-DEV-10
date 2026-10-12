import re

import tiktoken
import torch
from torch._dynamo.polyfills.itertools import cycle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import generate, get_device


class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []

        # 构建个位数相加的问题集合
        for i in range(1, 10):
            for j in range(1, 10):
                # 问题
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                # 标准答案
                ground_truth = i + j
                # (问题， 标准答案)
                self.data.append((prompt, ground_truth))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batch(self, prompts, responses, device):

        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
            response_ids = self.tokenizer.encode(response, allowed_special={"<|endoftext|>"})

            ids = prompt_ids + response_ids
            # 提示词部分的掩码为0，模型不全部分的掩码为1
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        # all_ids 中的数据长度可能不一样，所以需要对齐长度
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)  # 0作为pad token id
            padded_masks.append(mask + [0] * pad_len)  # 0作为填充符对应的掩码

        padded_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        padded_masks = torch.tensor(padded_masks, dtype=torch.long, device=device)

        return padded_ids, padded_masks

#############################################################################################################################
def calculate_reward(ground_truth, response):
    """ 奖励函数， 用来计算策略模型 （sft后的模型） 的回答（不全） 是否正确"""
    """ ground_truth： 正确答案，response： 模型的回答"""

    try:
        matches = re.findall(r"(-?\d+)", response)  # 匹配可正可负的数字
        if matches:
            predicted = int(matches[-1])  # 获取最后一个匹配的数值
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


def generate_group(
        model,
        tokenizer,
        prompts,
        ground_truths,
        group_size
):
    """ 针对一个问题，产生G条回答 """

    all_prompts = []     # 保存所有问题的数组
    all_responses = []   # 保存所有模型回答的数组
    all_advantages = []  # 保存所有回答的组内优势的数组

    for prompt, ground_truth in zip(prompts, ground_truths):

        # 每一条回答
        responses = []  # 保存的是prompt对应的group_size条回答
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature = 1.0)
            response = full_text[len(prompt):]
            responses.append(response)


        # 每一条回答回报
        rewards = torch.tensor([calculate_reward(ground_truth, response) for response in responses])

        # 每一条回答的组内优势
        advantages = rewards - rewards.mean()

        for response, advantage in zip(rewards, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, all_advantages

def compute_log_probs(model, ids):
    """
        model: 计算动作的对数概率的模型，ids是模型采样的轨迹（提示词+补全）
    """

    logits = model(ids)

#######################################################################################################################
# 配置
device = get_device()
sft_model_path = "./sft_model.pth"
grpo_model_save_path = "./grpo_model.pth"

# 超参数
learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2
eval_interval = 10                   # 每隔10个iter评估一次模型算数学题的能力
epsilon = 0.2                        # ϵ，裁剪的超参数
group_size = 8                       # 每个数学题，模型生成8条回答
batch_size = 4                       # 一批4个数学题

# 分词器、模型、优化器
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2(
    vocab_size=50257,
    max_context_len=256,
    embed_dim=384,
    n_head=6,
    n_layer=6,
    ff_dim=4 * 384,
    dropout_rate=0.1,
)
model.load_state_dict(torch.load(sft_model_path))
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# 旧策略π_θ_old
old_model = model.clone()
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)


# 循环训练
accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    # 获取一批问题和正确答案
    prompts, ground_truths = next(data_iter)

    # 将π_θ设置为π_θ_old
    old_model.load_state_dict(model.state_dict())

    # 1.使用旧策略采样轨迹，并计算组内有事
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts,ground_truths,group_size
    )
    # TODO 这一步是多余的吗
    all_advantages =  all_advantages.to(device)
    # 将采样的轨迹整理成批次数据
    ids, masks = dataset.get_batch(all_prompts, all_responses, device=device)

    # 2. 使用采样好的轨迹更新多次策略模型
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grop_loss(model, old_model, ids, masks, all_advantages, epsilon)



def grop_loss(model, old_model, ids, masks, advantages, epsilon):
    """ 计算GRPO损失函数 """

    log_probs = compute_log_probs(model, ids)


def compute_log_probs(model, ids):
    """ 计算动作对数概率， ids是模型采样的轨迹（提示词+补全） """
    logits = model(ids)   # (B,T,V)

    # 去掉最后一个位置的输出（它的预测目标不在序列内）
    log_dist = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B,T-1,V) 整个词表上的对数概率分布

    # 去掉第一个输入（第一个 token 没有预测目标）
    labels = ids[:, 1:]  # (B,T-1)

    # 从分布中取出"实际发生的动作"对应的对数概率
    action_log_probs = torch.gather(
        log_dist,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B,T-1)

    return action_log_probs