"""summary
DQN步骤：

- 记忆池里的数据样式
- CartPole-v0的状态由4位实数编码表示，所以第一层网络是4->50
"""
# %%
from abc import ABC
from numpy.lib.function_base import _quantile_dispatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
from matplotlib import pyplot as plt
import sys

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.4                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency
MEMORY_CAPACITY = 512
DEVICE = 3   # 指定GPU
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = 4  # 4种候选的算子
# N_STATES = 30  # 30维决策变量
use_gpu = torch.cuda.is_available()
# %%


class Net(nn.Module, ABC):
    def __init__(self, inDim, outDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inDim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.out = nn.Linear(32, outDim)

    def forward(self, x):
        # return self.out(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action_value = self.out(x)
        return action_value


class DQN(object):
    def __init__(self, inDim, outDim):
        self.eval_net, self.target_net = Net(inDim, outDim), Net(inDim, outDim)
        # global N_STATES, N_ACTIONS
        self.N_STATES = inDim
        self.N_ACTIONS = outDim
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # memory是一个np数组，每一行代表一个记录，状态 动作 奖励 新的状态
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        if use_gpu:
            self.eval_net, self.target_net = self.eval_net.cuda(DEVICE), self.target_net.cuda(DEVICE)
            self.loss_func = self.loss_func.cuda(DEVICE)

    def choose_action(self, x):
        # x: a game state
        # 在前面多加一维，可能是一批数据的意思
        # 返回的是0-1动作整数编码
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if use_gpu:
            x = x.cuda(DEVICE)
        # input only one sample
        # if np.random.uniform() < EPSILON:   # greedy
        if np.random.uniform() < 2:   # greedy
            actions_value = self.eval_net.forward(x)  # shape=(1,action)
            if use_gpu:
                actions_value = actions_value.cpu()
            actions_value = actions_value.detach().numpy()
            # print(actions_value)
            # action = np.argmax(actions_value[0])  # 选择回报最大的动作
            # print(action)
            # return action
            actions_value[actions_value <= 0] = 0.001  # 不能有负概率
            # actions_value = actions_value / np.sum(actions_value)  # 归一化
            # 计算排名
            argsort_ = self.N_ACTIONS - 1 - np.argsort(np.argsort(actions_value[0]))
            # 以系数c拉大概率差距
            # c = 0.5
            # for i in range(self.N_ACTIONS):
            # actions_value[0][i] = actions_value[0][i] * c**argsort_[i]
            # 手动设计概率
            probability_value = np.array([[70, 28, 10, 8, 5, 5]])
            probability_value = probability_value / np.sum(probability_value)
            actions_value = probability_value[:, argsort_]

            actions_value = actions_value / np.sum(actions_value)

            # 按照概率取样

            try:
                action = np.random.choice(self.N_ACTIONS, size=1, p=actions_value[0])[0]
            except:
                print(actions_value)
                action = np.random.randint(0, self.N_ACTIONS)
            # print(actions_value, action)
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)   # [0,N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # 数组合并，a和r也新建个数组
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        # 每隔一定步骤，更新target net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1].astype(int))  # 动作是int型
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        if use_gpu:
            b_s = b_s.cuda(DEVICE)
            b_a = b_a.cuda(DEVICE)
            b_r = b_r.cuda(DEVICE)
            b_s_ = b_s_.cuda(DEVICE)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        q_target = b_r
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    N_ACTIONS = 4
    N_STATES = 30
    BATCH_SIZE = 16
    MEMORY_CAPACITY = 40
    CountOpers = np.zeros(N_ACTIONS)
    PopCountOpers = []
    dqn = DQN(N_STATES, N_ACTIONS)
    s = np.random.uniform(-1, 1, N_STATES)
    s_ = np.random.uniform(-1, 1, N_STATES)

    print('\nCollecting experience...')
    for i_episode in range(1, 801):
        a = dqn.choose_action(s)
        # if i_episode > 400:
        #     r = -a
        # else:
        #     r = a
        r = a
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > 50:
            dqn.learn()
        CountOpers[a] += 1
        if i_episode % 5 == 0:
            print(i_episode, ' ', a)
            PopCountOpers.append(CountOpers)
            CountOpers = np.zeros(N_ACTIONS)

    PopCountOpers = np.array(PopCountOpers)
    for i in range(N_ACTIONS):
        plt.plot(PopCountOpers[:, i], '.', label=str(i))
    plt.legend()
    plt.show()

    sys.exit(0)


"""
    while True:
        # env.render()
        a = dqn.choose_action(s)

        # take action
        # s_, r, done, info = env.step(a)
        s_ = np.random.uniform(-1,1,N_STATES)

        # modify the reward
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        # r = a
        r = -(np.abs(a-2))
        r += 2
        r = np.random.normal(r,3,1)[0]

        dqn.store_transition(s, a, r, s_)
        # print(s,a,r,s_)
        # sys.exit(0)
        line = 20

        ep_r += r
        # if dqn.memory_counter > MEMORY_CAPACITY:
        if dqn.memory_counter > 20:
            dqn.learn()
            # print(a)
            # if ep_r>line:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', round(ep_r, 2))

        if ep_r>line:
            break
        s = s_

"""
