import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym
import sys
import os
import time
import datetime
sys.path.append(os.path.abspath('../'))
from utils.logger import Logger, TorchLogger
from utils.plotter import plot, make_gif
from model import QNet, ReplayBuffer
import argparse
import ast
import yaml
with open('config.yaml','r') as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser('Command line arguments')
parser.add_argument('env', type = str, default = 'CartPole-v1', help = 'Environment name for gym')
parser.add_argument('test', type = str, default = 'False', help = 'Run in eval mode')
args = parser.parse_args()
args.test = ast.literal_eval(args.test)
print(args)

if __name__ == '__main__':
    #Setup environment
    env_name = args.env
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env, 100)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        print('Environment not supported')
        sys.exit(1)

    total_ep = int(75)
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n

    q_net = QNet(obs_space_dims, action_space_dims)
    target_net = QNet(obs_space_dims, action_space_dims)

    #setup loggers
    exp_name = 'dqn' + str(datetime.date.today()) + str(int(time.time()))
    q_logger = Logger(dir = 'logs/', filename = 'q_net.txt', exp_name = exp_name)
    q_tlogger = TorchLogger(dir = 'logs/', 
                        exp_name = exp_name,
                        dict = {'model': q_net._policy_state_dict})
    target_logger = Logger(dir = 'logs/', filename = 'target_net.txt', exp_name = exp_name)
    target_tlogger = TorchLogger(dir = 'logs/', 
                        exp_name = exp_name,
                        dict = {'model': target_net._policy_state_dict})
    replay_buffer = ReplayBuffer(capacity = config['buffer_size'])

    reward_over_episodes = []

    ###Train Mode

    ###Eval Mode

    ## replay buffer
    ## add support for double dqn
    ## add support for duelling dqn