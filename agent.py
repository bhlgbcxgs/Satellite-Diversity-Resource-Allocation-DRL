import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, init_params):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        nn.init.constant_(self.linear3.bias.data, init_params['bias'])  #0.2 #0.1
        nn.init.uniform_(self.linear3.weight.data, init_params['weight_min'], init_params['weight_max'])

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_actions, init_params):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, num_actions)

        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.constant_(self.linear3.bias.data, init_params['bias'])
        nn.init.uniform_(self.linear3.weight.data, init_params['weight_min'], init_params['weight_max'])

    def forward(self, state, angle, ac):
        x = torch.cat([state, angle, ac], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        Q = self.linear3(x)

        return Q

    def select_action(self, state, angle, ac):
        with torch.no_grad():
            Q = self.forward(state, angle, ac)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()



class Multiplier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, init_params):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        nn.init.constant_(self.linear3.bias.data, init_params['bias'])
        nn.init.uniform_(self.linear3.weight.data, init_params['weight_min'], init_params['weight_max'])

    def forward(self, a):
        x = F.relu(self.linear1(a))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        net_config = getattr(self, 'net_config', {})

        s_dim = self.env.satellite_num * 2
        a_dim = self.env.satellite_num
        self.actor = Actor(s_dim, 64, 64, a_dim, init_params=net_config.get('actor'))  # 输入卫星仰角、CSI--输出功率分配
        self.multiplier = Multiplier(s_dim, 64, 64, 1, init_params=net_config.get('multiplier'))  # 对actornet输出的约束--功率分配和小于等于1
        self.Qnet = DQN(3, 256, 256, 13, init_params=net_config.get('dqn'))  # 输入卫星仰角&SNR 输出选择13种调制编码方式的概率（SNR=CSI*功率分配）
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.multiplier_optim = optim.Adam(self.multiplier.parameters(), lr=self.multiplier_lr)
        self.Qnet_optim = optim.Adam(self.Qnet.parameters(), lr=self.mcchooser_lr)

        self.buffer = []

    def act(self, s0, step):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).detach().squeeze().numpy()
        if step < self.r_1:
            sigma = self.initial_sigma
        elif step < self.r_2:
            sigma = self.initial_sigma - self.initial_sigma * (step - self.r_1) / (self.r_2 - self.r_1)
        else:
            sigma = 0
        epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * np.exp(-1 * step / 10000)
        mc0 = []
        for i in range(len(a0)):
            a0[i] = max(0, a0[i] + np.random.randn() * sigma)
            if random.random() > epsilon:
                mc = self.Qnet.select_action(s0.squeeze()[i].unsqueeze(0).unsqueeze(0),
                                             s0.squeeze()[i + self.env.satellite_num].unsqueeze(0).unsqueeze(0),
                                             torch.tensor(a0[i]).unsqueeze(0).unsqueeze(0))
            else:
                mc = random.randint(0, 12)
            mc0.append(mc)

        return a0, mc0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self, step):  # a:sampled
        if len(self.buffer) < self.batch_size:
            j_loss = torch.tensor(0, dtype=torch.float)
            return j_loss

        samples = random.sample(self.buffer, self.batch_size)
        s, a, mc, r = zip(*samples)
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(np.array(a), dtype=torch.float)
        mc = torch.tensor(np.array(mc), dtype=torch.float)
        r = torch.tensor(np.array(r), dtype=torch.float)
        s_t = torch.transpose(s, dim0=1, dim1=0)
        a_t = torch.transpose(a, dim0=1, dim1=0)
        mc_t = torch.transpose(mc, dim0=1, dim1=0)
        r_t = torch.transpose(r, dim0=1, dim1=0)


        def q_value_cal():
            q = torch.tensor([])
            for i in range(self.env.satellite_num):
                st = torch.unsqueeze(s_t[i], 1)
                anglet = torch.unsqueeze(s_t[i + self.env.satellite_num], 1)
                at = torch.unsqueeze(torch.transpose(self.actor(s), dim0=1, dim1=0)[i], 1)
                p = self.Qnet(st, anglet, at).gather(1, mc_t[i].unsqueeze(1).long())
                q = torch.cat([q, p], dim=1)
            sum = torch.sum(q, dim=1).unsqueeze(1)
            return sum

        def actor_learn():
            loss = - torch.mean(q_value_cal() - self.multiplier(s) * (
                    torch.sum(self.actor(s), dim=1) - 1).unsqueeze(1))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def multiplier_learn():
            loss = - torch.mean(self.multiplier(s) * (torch.sum(self.actor(s), dim=1) - 1).unsqueeze(1))
            self.multiplier_optim.zero_grad()
            loss.backward()
            self.multiplier_optim.step()

        def mc_learn():
            loss = 0
            for i in range(self.env.satellite_num):
                y_true = r_t[i]
                st = torch.unsqueeze(s_t[i], 1)
                anglet = torch.unsqueeze(s_t[i+self.env.satellite_num], 1)
                at = torch.unsqueeze(a_t[i], 1)
                y_pred = torch.squeeze(self.Qnet(st, anglet, at).gather(1, mc_t[i].unsqueeze(1).long()))
                loss_fn = nn.MSELoss()
                loss_i = loss_fn(y_pred, y_true)
                loss = loss + loss_i
                self.Qnet_optim.zero_grad()
                loss_i.backward()
                self.Qnet_optim.step()

            return loss

        if step % 2 == 0:
            actor_learn()
            multiplier_learn()
        j_loss = mc_learn()

        return j_loss
