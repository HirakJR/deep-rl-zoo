import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath('../'))
from utils.logger import TorchLogger
import yaml

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class GaussianPolicyNet(nn.Module):
    '''
    Parameterized Policy Network (Mapping from State -> Action)
    For Continuous action spaces
    Input: (obs_dim, action_dim)
    Output: (mean, std_dev) of a Normal Distribution from which action is to be sampled
    '''
    def __init__(self, obs_dim: int, action_dim: int):
        '''
        Constructor for the parameterized Policy Network.
        
        Arguments:
            obs_dim (int): Dimension of observation space;
            action_dim (int): Dimension of action space
        
        '''
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim_1 = self.obs_dim * 8
        self.hidden_dim_2 = self.obs_dim * 16

        self.shared_net = nn.Sequential(
                                nn.Linear(self.obs_dim, self.hidden_dim_1),
                                nn.Tanh(),
                                nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
                                nn.Tanh()
        )

        self.mean_branch = nn.Sequential(
                                nn.Linear(self.hidden_dim_2, self.action_dim)
        )
        self.std_dev_branch = nn.Sequential(
                                nn.Linear(self.hidden_dim_2, self.action_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns mean and standard deviation of a distribution from where action is to be sampled.
        Input:
            x: Observation
        Ouput:
            mean: Tensor of means from where action is to be sampled
            std_dev: Tensor of standard deviation from where action is to be sampled
        '''
        shared_net = self.shared_net(x.float())
        action_means = self.mean_branch(shared_net)
        action_std_devs = torch.log(1 + torch.exp(self.std_dev_branch(shared_net)))

        return action_means, action_std_devs

class CategoricalPolicyNet(nn.Module):
    '''
    Parameterized Policy Network (Mapping from State -> Action)
    For Discrete action spaces
    Input: (obs_dim, action_dim)
    Output: (mean, std_dev) of a Normal Distribution from which action is to be sampled
    '''
    def __init__(self, obs_dim: int, action_dim: int):
        '''
        Constructor for the parameterized Policy Network.
        
        Arguments:
            obs_dim (int): Dimension of observation space;
            action_dim (int): Dimension of action space
        
        '''
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim_1 = self.obs_dim * 8
        self.hidden_dim_2 = self.obs_dim * 16

        self.net = nn.Sequential(
                                nn.Linear(self.obs_dim, self.hidden_dim_1),
                                nn.Tanh(),
                                nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
                                nn.Tanh(),
                                nn.Linear(self.hidden_dim_2, self.action_dim)
        )

        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns values sampled from a Categorical Distribution from where action is to be sampled.
        Input:
            x: Observation
        Ouput:
            mean: Tensor of means from where action is to be sampled
            std_dev: Tensor of standard deviation from where action is to be sampled
        '''
        logits = self.net(x.float())
        return Categorical(logits = logits)
        

class VPG:
    '''
    Vanilla Policy Gradient
    More accurately,the REINFORCE or Monte Carlo Policy Gradient algorithm.
    '''
    def __init__(self,obs_dim: int, action_dim: int, env: gym.Env):
        '''
        Solves task using the REINFORCE or Monte Carlo Policy Gradient algorithm
        Arguments:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
        '''
        continuous = None
        if isinstance(env.action_space,gym.spaces.Discrete):
            continuous = False
        elif isinstance(env.action_spac, gym.spaces.Box):
            continuous = True
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env = env
        
        #Hyperparameters
        self.lr = config['lr']
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']

        #bookeeping for policy update later
        self.log_probs = list()
        self.rewards = list()
        if continuous:
            self.policy = GaussianPolicyNet(self.obs_dim, self.action_dim).to(device=device)
        else:
            self.policy = CategoricalPolicyNet(self.obs_dim, self.action_dim).to(device=device)


        self.optimizer = AdamW(params=self.policy.parameters(),
                               lr = self.lr)
        self._policy_state_dict = self.policy.state_dict()
    
    def sample_action(self, obs: np.ndarray) -> float:
        '''
        Return action using the Policy Network conditioned on the observation
        '''
        obs = torch.tensor(np.array(obs)).to(device)
        if isinstance(self.env.action_space, gym.spaces.Box):
            mean, std_dev = self.policy(obs)
            dist = Normal(mean[0] + self.epsilon, std_dev[0] + self.epsilon)
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            dist = self.policy(obs)
        
        action = dist.sample().to(device)
        self.log_probs.append(dist.log_prob(action).to(device))

        action = action.cpu().numpy().squeeze()

        return action
    
    def update(self):
        '''
        Update the Policy net
        '''
        self.policy.train()
        curr_return = 0
        returns = list()

        for reward in self.rewards[::-1]:
            curr_return = reward + self.gamma * curr_return
            self.rewards.insert(0, curr_return) 
        
        returns = torch.tensor(returns).to(device)

        loss = 0
        for log_prob, ret in zip(self.log_probs, self.rewards):
            loss += (-1) * log_prob.mean() * ret
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #empty cache
        self.log_probs = []
        self.rewards = []
    
    def eval(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

