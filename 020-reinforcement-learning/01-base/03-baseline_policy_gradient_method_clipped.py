from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

@dataclass
class Episode:
    """ 一条完整轨迹的打包记录 """
    states: torch.Tensor # 一条轨迹的状态：[S_0, S_1, ..., S_T]
    actions: torch.Tensor # 动作：[A_0, A_1, ..., A_T]
    _return: float # G(τ) = R_O + R_1 + .... + R_T
    old_log_probs: torch.Tensor # [logπ_θ_old(A_0|S_0), ...]

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        #4指的是:状态是4个浮点数：[推车的位置，推车的速度，木杆的角度，木杆的角速度]
        self.l1 = nn.Linear(4, 128)

        #2指的是2个动作: 向左推和向右推
        self.l2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        logits = self.l2(x)
        probs = F.softmax(logits, dim=-1)

        return probs

class Agent:

    def __init__(self):
        self.pi = PolicyNet() # 初始化一个策略神经网络π_θ
        self.lr_pi = 0.001    # 学习率
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr = self.lr_pi)



    def get_action(self, state):
        """ 根据状态采取动作 """
        # (4,) -> (1, 4)
        state = torch.tensor(state).unsqueeze(0)
        # 概率分布 (1,2) -> (2,)
        probs = self.pi(state)

        # 使用概率分布来抽样出一个动作（multinomial 使用probs创建一个概率分布， num_samples=1 表示采样一个动作）
        action = torch.multinomial(probs, num_samples=1).item()

        return action, probs



    def rollout(self, env):
        """ 采样： 在环境env中采样一条轨迹
            τ = (S_0, A_0, R_0, S_1, A_1, R_1, ..., S_{T+1})
        """

        # 开始一个新的轨迹
        state, _ = env.reset()   # state.shape(4,)


        # 轨迹信息
        states = []
        actions = []
        rewards = []

        # 是否结束
        done = False
        while not done:
            # 选择一个动作A_t
            action, _ = self.get_action(state)

            # 执行动作，返回的是 S_{t+1} , R_t, 是否失败， 是否玩满500步被截断， 调试信息
            next_state, reward, terminated, truncated, _ = env.step(action)


            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # 状态转义
            state = next_state


        states = torch.tensor(states) #(T,4)
        actions = torch.tensor(actions).view(-1,1) # (B,) --> (B,1)
        _return = sum(rewards)

        # 计算就策略采样的轨迹的每个动作的对数概率
        with torch.no_grad():
            old_log_probs = torch.log(
                torch.gather(
                    self.pi(states),
                    dim=-1,
                    index=actions
                )
            )

        return Episode(
            states = states,
            actions = actions,
            _return = _return,
            old_log_probs = old_log_probs
        )


    def update(self, episodes, avg_return):
        n_update_per_generation = 4 # 旧策略采样的轨迹更新4次新策略
        for _ in range(n_update_per_generation):
            objs = []
            for episode in episodes:

                # [logπ_θ(A_0|S_0), logπ_θ(A_1|S_1), ..., logπ_θ(A_T|S_T)]
                log_action_probs = torch.log(torch.gather(
                    self.pi(episode.states),  #一条轨迹每一步概率分布(T,2)
                    dim=-1,                   # 在最后一维(概率分布那一维)上取值
                    index=episode.actions     # 每步实际选中的动作编号
                ))

                # 比值ρ_t
                ratio = torch.exp(log_action_probs - episode.old_log_probs)

                # 裁剪后的比值clip(ratio, 1-0.2, 1+0.2)
                clipped_ratio = torch.clamp(ratio,1-0.2, 1+0.2)

                # 组内优势
                advantage = episode._return - avg_return

                # Σ_{t=0}^T{G(τ) - 一组轨迹回报的均值作为基线b(S_t) logπ_θ(A_t|S_t)}
                # 将sum改为mean
                # obj = torch.sum(log_action_probs) * (episode._return - avg_return)
                obj = torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))
                objs.append(obj)

            loss = - sum(objs) / len(objs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

env = gym.make("CartPole-v1")
agent = Agent()

# print(agent.rollout(env))


# 记录轨迹回报的数组
returns = []
max_steps_list = []
for iteration in range(1000): # 迭代1000次, 每次迭代 = 采样n条轨迹 + 更新一次策略
    # 1. 采样n条轨迹
    n = 8
    episodes = []
    for _ in range(n):
        episode = agent.rollout(env)
        episodes.append(episode)

    # 2. 计算n条轨迹的回报的平均值
    avg_return = sum([episode._return for episode in episodes]) / len(episodes)
    returns.append(avg_return)


    # 4. 使用采样的轨迹更新策略模型
    agent.update(episodes, avg_return)


    max_steps_list.append(max(len(ep.actions) for ep in episodes))

    if (iteration+1) % 100 == 0:

        max_steps = 0
        for episode in episodes:
            max_steps = max(max_steps, len(episode.actions))

        print(f"Iteration: {iteration+1}, Return: {avg_return}, Max_steps: {max_steps}")



fig = plt.figure()
plt.plot(returns, label="Return")
plt.plot(max_steps_list, label="Max_steps")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Return")
plt.title("CartPole-v1")
fig.savefig("baseline_clipped_pgm.pdf")
plt.show()













