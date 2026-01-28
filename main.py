import torch
import csv
from agent import Agent
from environment import EnvWF
import numpy as np
from config import Config


if __name__ == '__main__':

    env = EnvWF(snr_norm=Config.SNR_NORM_FACTOR, angle_norm=Config.ANGLE_NORM_FACTOR)
    agent_params = Config.TRAIN_PARAMS.copy()
    agent_params['env'] = env
    agent_params['net_config'] = Config.AGENT_INIT

    agent = Agent(**agent_params)

    eval_every = 500
    cum_reward = 0
    cum_loss = 0
    cum_constraint = 0

    average_reward_list = []
    average_loss_list = []
    average_constraint_list = []

    for episode in range(100000):
        s = env.ge_rnd_state(episode)
        a, mc = agent.act(s, episode)
        effi, ber = env.step(s, a, mc)
        agent.put(s, a, mc, effi)
        loss = agent.learn(episode)
        r = np.sum(np.array(effi))
        cum_reward = cum_reward + r
        a_re = []
        for i in range(len(a)):
            a_re.append(min(1, a[i]))
        g = np.sum(np.array(a_re)) - 1
        if g > 0:
            cum_constraint = cum_constraint + g
        cum_loss = cum_loss + loss.detach().numpy()

        if (episode + 1) % eval_every == 0:

            average_reward_list.append(cum_reward / eval_every)
            average_loss_list.append(cum_loss / eval_every)
            average_constraint_list.append(cum_constraint)

            cum_reward = 0
            cum_loss = 0
            cum_constraint = 0

            with open('./results/data/average_reward.csv', 'w', newline='') as logfile:
                wr = csv.writer(logfile)
                wr.writerow(average_reward_list)
            with open('./results/data/average_loss.csv', 'w', newline='') as logfile:
                wr = csv.writer(logfile)
                wr.writerow(average_loss_list)
            with open('./results/data/average_constraint.csv', 'w', newline='') as logfile:
                wr = csv.writer(logfile)
                wr.writerow(average_constraint_list)

    # save the networks
    torch.save(agent.actor, "./results/model/actor_model.pt")
    torch.save(agent.Qnet, "./results/model/Qnet_model.pt")
    torch.save(agent.multiplier, "./results/model/multiplier_model.pt")









