import gym
import pickle
import random
import numpy as np
import torch
from torch import nn
import os
import time
from itertools import count

device = "cuda" if torch.cuda.is_available() else "cpu"
action_dimension = {'BoxingNoFrameskip-v4': 18, 'PongNoFrameskip-v4': 6,
                    'BreakoutNoFrameskip-v4': 4, 'BowlingNoFrameskip-v4': 6,
                    'BattleZoneNoFrameskip-v4': 18, 'AssaultNoFrameskip-v4': 7}


class Buffer:
    def __init__(self, n):
        self.buffer = []
        self.n = n

    def store(self, sars):
        if len(self.buffer) > self.n:
            self.buffer.pop(0)
        self.buffer.append(sars)

    def sample(self, m):
        return random.sample(self.buffer, min(m, len(self.buffer)))


class DQNet(nn.Module):

    def __init__(self, dueling, n0, n1, n2, n3, n4):
        super(DQNet, self).__init__()
        self.dueling = dueling
        self.action_dimension = n4
        if not dueling:
            self.net = nn.Sequential(
                nn.Linear(n0, n1),
                nn.ReLU(),
                nn.Linear(n1, n2),
                nn.ReLU(),
                nn.Linear(n2, n3),
                nn.ReLU(),
                nn.Linear(n3, n4),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n0, n1),
                nn.ReLU(),
                nn.Linear(n1, n2),
                nn.ReLU(),
                nn.Linear(n2, n3),
                nn.ReLU(),
            )
            self.value = nn.Linear(n3, 1)
            self.advantage = nn.Linear(n3, n4)

    def forward(self, S):
        if not self.dueling:
            return self.net(S)
        else:
            h = self.net(S)
            v = self.value(h)
            a = self.advantage(h)
            a = a - a.mean(dim=1).reshape((-1, 1))
            return v + a

    def get_action(self, s, eps):
        if eps == 0 or np.random.random() > eps:
            self.eval()
            a = torch.argmax(self.forward(s), axis=1).reshape((-1, 1))
        else:
            a = np.random.randint(0, self.action_dimension)

        return a


class DDQN():
    def __init__(self, env='BoxingNoFrameskip-v4', lr_start=5e-5, lr_end=1e-6, batch_size=128, eps_start=0.2, eps_end=0.05,
                 gamma=0.99, weight_decay=1e-4, total_step=10000000, save_step=500000, h=256,
                 buffer_size=4096, dueling=True, soft_update=False, update_step=4000, tau=0.05):
        self.env = gym.make(env, obs_type='ram')
        self.env_name = env

        # hyperparameters
        if env in action_dimension:
            self.action_dimension = action_dimension[env]
        else:
            raise NotImplemented
        self.buffer = Buffer(buffer_size)
        self.batch_size = batch_size
        self.total_step = total_step
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step = (eps_start - eps_end) / total_step
        self.gamma = gamma
        self.soft_update = soft_update
        self.update_step = update_step
        self.tau = tau
        self.dueling = dueling
        self.save_step = save_step

        # networks
        self.model_update = DQNet(dueling=dueling, n0=128, n1=h, n2=h, n3=h, n4=self.action_dimension).to(device)
        self.model_target = DQNet(dueling=dueling, n0=128, n1=h, n2=h, n3=h, n4=self.action_dimension).to(device)
        self.model_target.load_state_dict(self.model_update.state_dict())
        self.optimizer = torch.optim.Adam(self.model_update.parameters(), lr=lr_start, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1, end_factor=(lr_end/lr_start), total_iters=total_step
        )

        os.makedirs('pickles', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    def train(self):
        R_all = []
        LOSS = []
        eps = self.eps_start
        step = 0
        for episode in count():
            R_episode = 0

            s = torch.from_numpy(self.env.reset().reshape((1, -1))).float().to(device)

            for t in count():
                self.env.render()
                # get an action
                a = self.model_update.get_action(s, eps)
                # print(a)

                # interact with the environment
                s_, r, done, info = self.env.step(a)
                s_ = torch.from_numpy(s_.reshape((1, -1))).float().to(device)
                R_episode += r

                self.buffer.store([
                    s,
                    torch.Tensor([a]).reshape((1, 1)).long().to(device),
                    torch.Tensor([r]).reshape((1, 1)).float().to(device),
                    s_,
                    torch.Tensor([int(done)]).reshape((1, 1)).float().to(device),
                ])
                s = s_

                # update the network
                experience = self.buffer.sample(self.batch_size)                # batch x 5 (list)
                # print(experience)

                S_ = torch.concat([e[3] for e in experience])                   # batch x 128
                R = torch.concat([e[2] for e in experience])                    # batch x 1
                mask = torch.concat([e[4] for e in experience])                 # batch x 1

                A_ = self.model_update.get_action(S_, 0)    # batch x 1
                self.model_target.eval()
                out = self.model_target(S_)
                Q = torch.gather(out, 1, A_)

                Y = R + self.gamma * Q * (1 - mask)                             # batch x 1

                S = torch.concat([e[0] for e in experience])
                A = torch.concat([e[1] for e in experience])

                # train the network!
                self.model_update.train()
                out = self.model_update(S)
                out = torch.gather(out, 1, A)
                loss = self.criterion(out, Y)
                LOSS.append(float(loss))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update step information
                if done:
                    print(f"Episode {episode} finished after {t+1} steps, R: %.0f" % R_episode)
                    break
                if step == self.total_step: break

                step += 1
                self.scheduler.step()
                eps -= self.eps_step

                # update the target network
                if self.soft_update:
                    for update, target in zip(self.model_update.parameters(), self.model_target.parameters()):
                        target.data = self.tau * update.data + (1 - self.tau) * target.data

                elif step % self.update_step == 0:
                    self.model_target.load_state_dict(self.model_update.state_dict())

                if step % self.save_step == 0:
                    with open(f'pickles/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_loss_{step}.pickle', 'wb') as f:
                        pickle.dump(LOSS, f)
                    with open(f'pickles/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_reward_{step}.pickle', 'wb') as f:
                        pickle.dump(R_all, f)
                    with open(f'models/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_model_{step}.pickle', 'wb') as f:
                        pickle.dump(self.model_update.to('cpu'), f)
                        self.model_update.to(device)

            if step == self.total_step: break
            R_all.append(R_episode)

        self.env.close()
        pickle.dump(R_all, open(f'pickles/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_reward.pickle', 'wb'))
        pickle.dump(LOSS, open(f'pickles/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_loss.pickle', 'wb'))
        pickle.dump(self.model_update.to('cpu'), open(f'models/DDQN{"_dueling" if self.dueling else ""}_{self.env_name}_model.pickle', 'wb'))

    def eval(self, filename):
        with open(filename, 'rb') as f: self.model_update = pickle.load(f)
        for episode in range(100):
            R_episode = 0
            s = self.env.reset().reshape((1, -1))

            for t in count():
                self.env.render()
                # get an action
                a = self.model_update.get_action(torch.from_numpy(s).float().to(device), 0)
                # print(a)

                # interact with the environment
                s_, r, done, info = self.env.step(a)
                s_ = s_.reshape((1, -1))
                R_episode += r
                s = s_
                if done:
                    print(f"Episode {episode} finished after {t+1} steps, R: %.0f" % R_episode)
                    break

        self.env.close()


if __name__ == '__main__':
    ddqn = DDQN(env='BoxingNoFrameskip-v4')
    ddqn.train()

