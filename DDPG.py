import numpy as np
from torch import nn
import gym
import pickle
import time
import os
import matplotlib.pyplot as plt
from itertools import count
from utilities import *


class Critic(nn.Module):

    def __init__(self, n0, n1, n2, n3, n4):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n0, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.Linear(n3, n4),
        )

    def forward(self, x):
        return self.net(x)

    def get_q(self, s, a):
        self.train()
        input = torch.concat([s, a], dim=1)
        q = self.forward(input)

        return q


class Actor(nn.Module):
    def __init__(self, n0, n1, n2, n3, n4):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n0, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.Linear(n3, n4),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * 2

    def get_action(self, s):
        self.train()
        a = self.forward(s)

        return a


class DDPG():

    def __init__(self, env, lr_start=5e-5, lr_end=5e-7, batch_size=128, eps_start=0.2, eps_end=0.01, gamma=0.99,
                 weight_decay=1e-3, total_step=3000000, save_step=1000000, h=256, buffer_size=4096,
                 soft_update=False, update_step=1000, tau=0.05):
        self.env = gym.make(env)
        self.env_name = env

        # hyperparameters
        self.action_dimension = self.env.action_space.shape[0]
        self.state_dimension = self.env.observation_space.shape[0]
        self.transition_buffer = Buffer(buffer_size)
        self.policy_buffer = Buffer(buffer_size)
        self.batch_size = batch_size
        self.total_step = total_step
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step = (eps_start - eps_end) / total_step
        self.gamma = gamma
        self.soft_update = soft_update
        self.update_step = update_step
        self.tau = tau
        self.save_step = save_step

        # networks
        self.c_update = Critic(n0=(self.action_dimension+self.state_dimension), n1=h, n2=h, n3=h, n4=1).to(device)
        self.c_target = Critic(n0=(self.action_dimension+self.state_dimension), n1=h, n2=h, n3=h, n4=1).to(device)
        self.a_update = Actor(n0=self.state_dimension, n1=h, n2=h, n3=h, n4=self.action_dimension).to(device)
        self.a_target = Actor(n0=self.state_dimension, n1=h, n2=h, n3=h, n4=self.action_dimension).to(device)

        self.a_target.load_state_dict(self.a_update.state_dict())
        self.c_target.load_state_dict(self.c_update.state_dict())

        self.optimizer_c = torch.optim.Adam(self.c_update.parameters(), lr=lr_start, weight_decay=weight_decay)
        self.optimizer_a = torch.optim.Adam(self.a_update.parameters(), lr=lr_start, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler_c = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer_c, start_factor=1, end_factor=(lr_end/lr_start), total_iters=total_step
        )
        self.scheduler_a = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer_a, start_factor=1, end_factor=(lr_end/lr_start), total_iters=total_step
        )

        os.makedirs('pickles', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('figures', exist_ok=True)

    def train(self):

        R_all = []
        LOSS = []
        eps = self.eps_start
        step = 0

        for episode in count():
            R_episode = 0

            s = torch.from_numpy(self.env.reset().reshape((1, -1))).float().to(device)

            for t in count():

                # get an action
                a = self.a_update.get_action(s).cpu().detach().numpy()
                self.policy_buffer.store(s)
                # print(a)

                # interact with the environment (with noise)
                s_, r, done, info = self.env.step(np.ravel(a + np.random.randn(*a.shape) * eps))
                R_episode += r
                s_ = torch.from_numpy(s_.reshape((1, -1))).float().to(device)

                self.transition_buffer.store([
                    s,
                    torch.from_numpy(a).float().to(device),
                    torch.Tensor([r]).reshape((1, 1)).float().to(device),
                    s_,
                    torch.Tensor([int(done)]).reshape((1, 1)).float().to(device)
                ])
                s = s_

                # update the networks
                experience = self.transition_buffer.sample(self.batch_size)     # batch (list)
                # print(experience)

                S_ = torch.concat([e[3] for e in experience])                   # batch x state_dim
                R = torch.concat([e[2] for e in experience])                    # batch x 1
                mask = torch.concat([e[4] for e in experience])                 # batch x 1

                A_ = self.a_target.get_action(S_).detach()                      # batch x 1
                Q = self.c_target.get_q(S_, A_).detach()                        # batch x 1

                Y = R + self.gamma * Q * (1 - mask)                             # batch x 1

                S = torch.concat([e[0] for e in experience])
                A = torch.concat([e[1] for e in experience])

                X = torch.concat([S, A], axis=1)

                # update the critic
                self.c_update.train()
                out = self.c_update(X)
                loss = self.criterion(out, Y)
                LOSS.append(float(loss))

                self.optimizer_c.zero_grad()
                loss.backward()
                self.optimizer_c.step()

                # update the actor
                S = self.policy_buffer.sample(self.batch_size)
                S = torch.concat(S)
                A = self.a_update.get_action(S)
                q = self.c_update.get_q(S, A)
                loss = - q.mean()

                self.optimizer_c.zero_grad()
                self.optimizer_a.zero_grad()
                loss.backward()
                self.optimizer_a.step()

                # update step information
                if done:
                    print(f"Episode {episode} finished after {t+1} steps, R: %.0f" % R_episode)
                    break
                if step == self.total_step: break

                step += 1
                self.scheduler_c.step()
                self.scheduler_a.step()
                eps -= self.eps_step

                # update the target networks
                if self.soft_update:
                    for update, target in zip(self.c_update.parameters(), self.c_target.parameters()):
                        target.data = self.tau * update.data + (1 - self.tau) * target.data

                    for update, target in zip(self.a_update.parameters(), self.a_target.parameters()):
                        target.data = self.tau * update.data + (1 - self.tau) * target.data

                elif step % self.update_step == 0:
                    self.c_target.load_state_dict(self.c_update.state_dict())
                    self.a_target.load_state_dict(self.a_update.state_dict())

                if step % self.save_step == 0:
                    with open(f'models/DDPG_{self.env_name}_critic_{step}.pickle', 'wb') as f:
                        pickle.dump(self.c_update.to('cpu'), f)
                        self.c_update.to(device)
                    with open(f'models/DDPG_{self.env_name}_actor_{step}.pickle', 'wb') as f:
                        pickle.dump(self.a_update.to('cpu'), f)
                        self.a_update.to(device)

            if step == self.total_step: break
            R_all.append(R_episode)

        self.env.close()
        pickle.dump(R_all, open(f'pickles/DDPG_{self.env_name}_reward.pickle', 'wb'))
        pickle.dump(LOSS, open(f'pickles/DDPG_{self.env_name}_loss.pickle', 'wb'))
        pickle.dump(self.c_update.to('cpu'), open(f'models/DDPG_{self.env_name}_critic.pickle', 'wb'))
        pickle.dump(self.a_update.to('cpu'), open(f'models/DDPG_{self.env_name}_actor.pickle', 'wb'))

        # plot the rewards curve
        R_smoothed = []
        moving_average(R_all, R_smoothed)
        color = 'darkorange'
        x = range(len(R_all))
        plt.plot(x, R_smoothed, color=color)
        plt.plot(x, R_all, alpha=0.2, color=color)
        plt.xlabel('Episode')
        plt.ylabel('Reward per episode')
        plt.title(f'Rewards curve of DDPG on {self.env_name}')
        plt.grid(which='both')
        plt.savefig(f'figures/ddpg_{self.env_name}.png', dpi=300)

    def eval(self, filename):
        with open(filename, 'rb') as f: self.a_update = pickle.load(f)

        for episode in range(1000):
            R_episode = 0
            s = self.env.reset().reshape((1, -1))

            for t in count():
                self.env.render()
                a = self.a_update.get_action(torch.from_numpy(s).float().to(device)).cpu().detach().numpy()
                s_, r, done, info = self.env.step(np.ravel(a))
                R_episode += r
                s = s_
                if done:
                    print(f"Episode {episode} finished after {t+1} time steps, R: %.0f" % R_episode)
                    break

        self.env.close()


if __name__ == '__main__':
    ddpg = DDPG('HalfCheetah-v2')
    tic = time.time()
    ddpg.train()
    print(f'Training time: {time.time()-tic}s.')

'''
some suggested games and their state, action space

HalfCheetah-v2: 17, 6,
Hopper-v2: 11, 3,
Humanoid-v2': 376, 17,
Ant-v2': 111, 8
'''