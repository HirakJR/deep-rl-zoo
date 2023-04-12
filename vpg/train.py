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
from model import VPG
import argparse
import ast

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


    total_ep = int(75)
    obs_space_dims = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_space_dims = env.action_space.n
    if isinstance(env.action_space, gym.spaces.Box):
        action_space_dims = env.action_space.shape[0]

    agent = VPG(obs_space_dims, action_space_dims,env)

    exp_name = 'vpg' + str(datetime.date.today()) + str(int(time.time()))
    logger = Logger(dir = 'logs/', filename = 'log.txt', exp_name = exp_name)
    tlogger = TorchLogger(dir = 'logs/', 
                        exp_name = exp_name,
                        dict = {'agent': agent._policy_state_dict})   


    reward_over_episodes = []

    ###Train Mode
    if not args.test:

        for ep in range(total_ep):
            obs, info = env.reset()
            done = False
            while not done:
                action = np.array(agent.sample_action(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                agent.rewards.append(reward)

                done = terminated or truncated

            log_dict = {'Episodic return': env.return_queue[-1][0],
                        'Average return': int(np.mean(env.return_queue)),
                        'Episode': ep}
            logger.log(log_dict)
            agent.update()
            if len(env.return_queue) > 2:
                if (env.return_queue[-1] > env.return_queue[-2]):
                    tlogger.save_best_checkpoint('vpg',env_name)

            #print to terminal, save checkpoint
            if ep % 250 == 0:
                    avg_reward = int(np.mean(env.return_queue))
                    print("Episode:", ep, "Average Reward:", avg_reward)
                    tlogger.save_checkpoint(step = ep)

        logger.dump_csv()
        plot(pd.read_csv(logger.dump_csv()), save = True, name = 'vpg' + str(env_name))

    ###Eval Mode
    if args.test:
        checkpoint = tlogger.load_checkpoint(f'runs/vpg/{env_name}.pt')
        agent.eval(checkpoint['agent'])
        total_ep = 50
        envv = gym.make(env_name, render_mode = 'rgb_array')
        envv = gym.wrappers.HumanRendering(envv)

        for ep in range(total_ep):
            obs, info = envv.reset()
            done = False
            while not done:
                action = np.array(agent.sample_action(obs))
                obs, reward, terminated, truncated, info = envv.step(action)
                done = terminated or truncated

