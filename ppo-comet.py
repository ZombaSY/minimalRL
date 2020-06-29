import gym
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.log import Log
from torch.distributions import Categorical
from utils.utils import save_model
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float, default=0.0005)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--lmbda', type=float, default=0.9)
parser.add_argument('--eps-clip', type=float, default=0.1)
parser.add_argument('--K-epoch', type=int, default=3)
parser.add_argument('--T-horizon', type=int, default=20)
parser.add_argument('--epsilon', type=float, default=0.1)
args = parser.parse_args()


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        self.device = torch.device('cuda')  # or cpu

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(args.K_epoch):
            td_target = r + args.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0

            # 이부분 이해가 잘...
            for delta_t in delta[::-1]:
                advantage = args.gamma * args.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-args.eps_clip, 1+args.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    log = Log(__file__[:-3])

    experiment = Experiment(api_key="F8yfdGljIExZoi73No4gb1gF5",
                            project_name="reinforcement-learning", workspace="zombasy")

    experiment.set_model_graph(model)

    for n_epi in range(2000):
        s = env.reset()
        done = False
        epsilon = max(0.01, args.epsilon - 0.01 * (n_epi / 200))
        while not done:
            for t in range(args.T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()

                coin = random.random()
                if coin < epsilon:
                    a = random.randint(0, 1)

                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r

                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            log.info("episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            experiment.log_metric('score', score / print_interval)
            experiment.log_metric('epsilon', epsilon)
            score = 0.0

        if n_epi % 500 == 0 and n_epi != 0:
            save_model(model, 'ppo', n_epi, experiment)

    env.close()


if __name__ == '__main__':
    main()
