import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.optim.adam import Adam
import math
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import PyFlyt.gym_envs
import pybullet
import pybullet_data

from gymnasium.vector import AsyncVectorEnv
import time

config = {
    'learning_rate'  : 0.0003,
    'gamma'           : 0.9,
    'lmbda'           : 0.9,
    'eps_clip'        : 0.2,
    'K_epoch'         : 10,
    'rollout_len'    : 1,
    'buffer_size'    : 4,
    'minibatch_size' : 32,
    "entropy_coeff": 0.001,
}

class RunningNormalizer:
    def __init__(self, shape, epsilon=1e-8):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)



class PPO_RND(nn.Module):
    def __init__(self, config):
        super(PPO_RND, self).__init__()
        self.data = []
        self.entropy_coeff = config['entropy_coeff']
        self.learning_rate = config['learning_rate']
        self.fc1   = nn.Linear(33,128)
        self.fc_mu = nn.Linear(128,7)
        self.fc_std  = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        self.fc_dv = nn.Linear(128,1)
        self.target_network = nn.Sequential(nn.Linear(33,128), nn.ReLU(), nn.Linear(128,1))
        self.prediction_network = nn.Sequential(nn.Linear(33,128), nn.ReLU(), nn.Linear(128,1))
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimization_step = 0
        self.buffer_size = config['buffer_size']
        self.minibatch_size = config['minibatch_size']
        self.rollout_len = config['rollout_len']
        self.eps_clip = config['eps_clip']
        self.gamma = config['gamma']
        self.lmbda = config['lmbda']
        self.K_epoch = config['K_epoch']

        self.normalizer = RunningNormalizer(shape=(33,))

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        raw_mu = self.fc_mu(x)

        mu = torch.zeros_like(raw_mu)

        control_indices = [0, 1, 2, 5, 6] 
        for idx in control_indices:
            mu[..., idx] = torch.tanh(raw_mu[..., idx])

        thrust_indices = [3, 4] 
        for idx in thrust_indices:
            mu[..., idx] = torch.sigmoid(raw_mu[..., idx])

        std = F.softplus(self.fc_std(x)) + 1e-6
        return mu, std


    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)

        return v
    def dv(self, x):
        x = F.relu(self.fc1(x))
        dv = self.fc_dv(x)

        return dv
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, r_batch, rnd_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], [],[]
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, rnd_lst,s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r[0]])
                    rnd_lst.append([r[1]])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                rnd_batch.append(rnd_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = (
                torch.tensor(np.array(s_batch), dtype=torch.float),
                torch.tensor(np.array(a_batch), dtype=torch.float) ,
                torch.tensor(np.array(r_batch), dtype=torch.float) ,
                torch.tensor(np.array(rnd_batch), dtype=torch.float),
                torch.tensor(np.array(s_prime_batch), dtype=torch.float) ,
                torch.tensor(np.array(done_batch), dtype=torch.float) ,
                torch.tensor(np.array(prob_a_batch), dtype=torch.float))
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, rnd_r,s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
                div_td_target = rnd_r + self.gamma * self.dv(s_prime) * done_mask
                div_delta = div_td_target - self.dv(s)
            delta = delta.numpy()
            div_delta = div_delta.numpy()
            div_advantage_lst=[]
            advantage_lst = []
            advantage = 0.0
            div_advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            for delta_t in div_delta[::-1]:
                div_advantage = self.gamma * self.lmbda * div_advantage + delta_t[0]
                div_advantage_lst.append([div_advantage])
            div_advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            div_advantage = torch.tensor(div_advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, rnd_r, s_prime, done_mask, old_log_prob, td_target, div_td_target, advantage, div_advantage))

        return data_with_adv


    def train_net(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            print("enter training")

            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, rnd_r, s_prime, done_mask, old_log_prob, td_target, div_td_target, advantage, div_advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    a = torch.squeeze(a, 2)
                    old_log_prob = torch.squeeze(old_log_prob, 2)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)

                    surr1 = ratio * (advantage+div_advantage)
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * (advantage+div_advantage)
                    loss = -torch.min(surr1, surr2) + (self.v(s) - td_target).pow(2).mean() + (self.dv(s)- div_td_target).pow(2).mean()
                    idx = torch.randint(0, s.size()[0], size=((s.size()[0])//8,))


                    self.normalizer.update(s_prime)
                    norm_s_prime = self.normalizer.normalize(s_prime)

                    with torch.no_grad():
                      target_output = self.target_network(norm_s_prime[idx])
                    predicted_output = self.prediction_network(norm_s_prime[idx])

                    rnd_loss = (target_output - predicted_output).pow(2).mean()

                    loss += rnd_loss
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1


def main():
    NUM_ENVS = 1
    # env = gym.make_vec("PyFlyt/Rocket-Landing-v4", num_envs=NUM_ENVS, vectorization_mode="async")
    env = gym.make("PyFlyt/Rocket-Landing-v4")

    model = PPO_RND(config)
    score, div_score, best_score = 0.0, 0.0, -10000

    print_interval = 20
    rollout = []

    writer = SummaryWriter(log_dir=f'runs/PPO_RND')

    for n_epi in range(500000):
        s, _ = env.reset()
        done = False
        count = 0
        episodic_score, episodic_int_score = 0.0, 0.0
        score=0.0
        st_time = time.time()
        while not done:
            for t in range(model.rollout_len):
                mu, std = model.pi(torch.from_numpy(s).float())

                dist = Normal(mu, std)
                a = dist.sample()
                action_mins = torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0])
                action_maxs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

                a = torch.clamp(a, action_mins, action_maxs)
                log_prob = dist.log_prob(a)
                s_prime, r, done, truncated, info = env.step(a.cpu().numpy())
                s_prime_tensor = torch.from_numpy(s_prime).float()
                model.normalizer.update(s_prime_tensor)
                norm_s_prime_tensor = model.normalizer.normalize(s_prime_tensor)
                with torch.no_grad():
                  int_r = (model.target_network(norm_s_prime_tensor)- model.prediction_network(norm_s_prime_tensor)).pow(2).mean()

                rollout.append((s, a, (r, int_r.item()*10), s_prime, log_prob.detach(), done))

                if len(rollout) == model.rollout_len:
                    model.put_data(rollout)
                    rollout = []
                    model.train_net()

                s = s_prime
                score += r
                episodic_score+=r
                div_score += int_r
                episodic_int_score+=int_r
                count += 1
            

        print(len(model.data), count)
        end_time = time.time()
        print("Total Time: ", end_time-st_time)

        score_val = score.mean().item()
        writer.add_scalar("Extrinsic Reward", episodic_score, n_epi)
        writer.add_scalar("Intrinsic_Reward", episodic_int_score, n_epi)




        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score_val, model.optimization_step))
        if n_epi > 10 and score_val > best_score:
            torch.save(model, "./Saved_Models_RND_DRL/best_model_ppo.pt")
            best_score = score_val
            print(f"new model saved. current best score: {best_score} ")
        if n_epi%100==0:
            torch.save(model, "./Saved_Models_RND_DRL/"+ str(n_epi)+"_model_ppo.pt")

    env.close()

if __name__ == '__main__':
    main()