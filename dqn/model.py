import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath('../'))
from utils.logger import TorchLogger
import yaml
from collections import deque, namedtuple
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class QNet(nn.Module):
    '''
    Parameterized Q fucntion, outputs Q values conditioned on input observation
    Defined only for Discrete action spaces
    Uses discrete observations instead of pixel due to limited compute
    Arguments:
        obs_dims (int): number of obs_dims 
        action_dims (int): number of action dimensions 
    '''
    def __init__(self, obs_dims: int, action_dims: int):
       super(QNet, self).__init__()

       self.obs_dims = obs_dims
       self.action_dims= action_dims

       self.net = nn.Sequential(
                                nn.Linear(self.obs_dims, 32),
                                nn.ReLU(),
                                nn.Linear(32, 64),
                                nn.ReLU(),
                                nn.Linear(64, 16),
                                nn.ReLU(),
                                nn.Linear(16, self.action_dims)
                                
       )
    def forward(self,x):
        logits = self.net(x)
        return Categorical(logits=logits)
    
#reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.transition = namedtuple('transition', (state, action, next_state, reward))
        self.replay_memory  = deque((), maxlen=capacity)
    
    def push(self,*args):
        self.replay_memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

