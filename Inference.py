import gym
from ppo import PPO
import torch
from torch.distributions import Categorical

def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    saved_model = torch.load('models/ppo_model9500.pt')

    model.load_state_dict(saved_model)

    while True:
        score = 0
        s = env.reset()

        for i in range(200):
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)

            model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
            s = s_prime

            score += r
            env.render()
            if done:
                break

        print('score = {}'.format(score))


if __name__ == '__main__':
    main()
