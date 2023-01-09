# import random
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

EPISODE_SIZE = 200
EPISODE_LENGTH = 200
LEARNING_RATE = 1e-2
EXPLORING_RATE = 0.9
EXPLORING_RATE_DECAY = 0.995
EXPLORING_RATE_MIN = 0.1
GAMMA = 0.95

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(epoch):
  # if epoch % 10 != 0:
  #   return max(LEARNING_RATE * pow(0.1, epoch - 1), 1e-4) 
  # return max(LEARNING_RATE * pow(0.1, epoch / 10), 1e-4)
  return max(0.01, min(0.5, 1.0 - math.log10((epoch+1)/25)))

def get_state(observation, state_bounds, observation_num):
  state = [0] * len(observation)
  for i, s in enumerate(observation):
    l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
    if s <= l:
      state[i] = 0
    elif s >= u:
      state[i] = observation_num[i] - 1
    else:
      state[i] = int(((s - l) / (u - l)) * observation_num[i])

  return tuple(state)

def plot_lr(all_lr, n_episodes, algo):
  plt.plot(list(range(n_episodes)), all_lr)
  plt.xlabel('episode')
  plt.ylabel('rewards')
  plt.ylim(0, 1+2)
  plt.title('Rewards over episodes ({})'.format(algo))
  plt.show()

def plot_rewards(rewards, n_episodes, algo):
  plt.plot(list(range(n_episodes)), rewards)
  plt.xlabel('episode')
  plt.ylabel('rewards')
  plt.ylim(0, EPISODE_LENGTH+5)
  plt.title('Rewards over episodes ({})'.format(algo))
  plt.show()

class Agent(nn.Module):
  def __init__(self, n_observations, n_actions):
    super(Agent, self).__init__()
    self.layer1 = nn.Linear(n_observations, 4)
    self.layer2 = nn.Linear(4, 24)
    self.layer3 =  nn.Linear(24, 24)
    self.layer4 = nn.Linear(24, n_actions)
    
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = F.relu(self.layer3(x))
    return self.layer4(x)

  def choose_action(self, exploring_rate, q_table, action_space, state):
    if np.random.random_sample() < exploring_rate:
      return action_space.sample()
    else:
      return np.argmax(q_table[state])


def main():
  env = gym.make('CartPole-v1')
  agent = Agent(env.observation_space.shape[0], env.action_space.n)
  all_rewards = []
  observations_num = (1, 1, 6, 3)
  actions_num = (2, )
  q_table = np.zeros(observations_num + actions_num)
  state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
  state_bounds[1] = [-0.5, 0.5]
  state_bounds[3] = [-math.radians(50), math.radians(50)]
  all_lr = []

  for i in range(EPISODE_SIZE):
    # exploring_rate = max(EXPLORING_RATE * pow(EXPLORING_RATE_DECAY, i), EXPLORING_RATE_MIN)
    exploring_rate = max(0.01, min(1, 1.0 - math.log10((i+1)/25)))
    learning_rate = max(0.01, min(0.5, 1.0 - math.log10((i+1)/25)))
    all_lr.append(exploring_rate)

    observation = env.reset()
    observation = np.asarray(observation[0])
    each_reward = 0
    state = get_state(observation, state_bounds, observations_num)
    for step in range(EPISODE_LENGTH):
      env.render()
      action = agent.choose_action(exploring_rate, q_table, env.action_space, state)
      observation, reward, done, _, info = env.step(action)
      each_reward += reward
      next_state = get_state(observation, state_bounds, observations_num)
      q_next_max = np.amax(q_table[next_state])
      q_table[state + (action,)] += learning_rate * (reward + GAMMA * q_next_max - q_table[state + (action,)])

      if done:
        # print('Episode finished after {} timesteps, total rewards {}'.format(step+1, each_reward))
        break
    all_rewards.append(each_reward)

  plot_rewards(all_rewards, EPISODE_SIZE, 'Q-table')
  plot_lr(all_lr, EPISODE_SIZE, 'Q-table')
  env.close()

if __name__ == '__main__':
  main()