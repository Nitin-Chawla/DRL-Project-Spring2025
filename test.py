import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np
from stable_baselines3 import PPO
import os
import torch
from PPO_DRL import PPO
from RND_DRL import PPO_RND
from RND_DRL import RunningNormalizer
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt

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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

env_id = "PyFlyt/Rocket-Landing-v4"
env = gym.make(env_id, max_displacement=200.0, render_mode='human')
# env = gym.make(env_id)
NUM_TEST_EPISODES = 1

# Load the trained model
# model = PPO.load("./ppo_pyflyt_best_model/best_model")

# model = torch.load("./Saved_Models_RND_DRL/best_model_ppo.pt", weights_only=False)

model = PPO(config)
model.load_state_dict(torch.load("./Saved_Models_PPO_DRL/best_model_ppo.pt"))
model.eval()

def test_model(model, env, num_episodes=NUM_TEST_EPISODES, render=False):
    total_rewards = []  #
    velocity_list = []
    steps_list = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        reward_list = []
        c = 0
        info = {'line_vel': [0, 0, 0]}
        while not done:
            c += 1
            # action, _states = model.predict(obs, deterministic=True)  # Get action from the model
            # obs, reward, done, _, info = env.step(action)

            action, std = model.pi(torch.from_numpy(obs).float())
            action_mins = torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0])
            action_maxs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            action = torch.max(torch.min(action, action_maxs), action_mins)

            obs, reward, done, _, info = env.step(action.detach().cpu().numpy())

            print(action, info['line_vel'], info['line_pos'])
            velocity_list.append(info['line_vel'][2]*-1)
            steps_list.append(c)

            episode_reward += reward
            reward_list.append(reward)
        total_rewards.append(episode_reward)
        print(c)
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} test episodes: {avg_reward}")

    plt.title("Velocity Plot")
    plt.xlabel("Steps")
    plt.ylabel("Vertical Velocity")
    plt.plot(steps_list, velocity_list)
    plt.show()
    plt.savefig("my_plot.png")


    return total_rewards, avg_reward


test_rewards, avg_test_reward = test_model(model, env, num_episodes=NUM_TEST_EPISODES)

np.save("test_rewards.npy", test_rewards)  # Save individual episode rewards
np.save("avg_test_reward.npy", avg_test_reward)  # Save the average test reward

print(f"Test Results - Average Reward: {avg_test_reward}")
