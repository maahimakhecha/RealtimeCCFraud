# src/rl_environment.py

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn

# Define the same network architecture as used in the supervised model.
class SupervisedNet(nn.Module):
    def __init__(self, input_dim):
        super(SupervisedNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class FraudDetectionEnv(gym.Env):
    """
    Custom Gym environment for fraud detection using PyTorch.
    
    Each episode corresponds to one transaction.
    The state is a vector of transaction features (default: 30 features).
    If a pretrained supervised model is provided (via pretrained_model_path),
    its fraud probability prediction is appended as an extra feature.
    
    Action space:
      0: Approve (do not flag fraud)
      1: Flag as fraud
    
    Reward:
      - If transaction is fraudulent (Class==1):
            +10 if action==1 (correct), -10 if action==0 (incorrect)
      - If transaction is legitimate (Class==0):
            +1 if action==0 (correct), -1 if action==1 (incorrect)
    """
    
    def __init__(self, feature_dim=30, pretrained_model_path=None):
        super(FraudDetectionEnv, self).__init__()
        self.feature_dim = feature_dim
        self.use_pretrained = False
        self.pretrained_model = None
        
        if pretrained_model_path is not None:
            self.pretrained_model = SupervisedNet(feature_dim)
            self.pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
            self.pretrained_model.eval()
            self.use_pretrained = True
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim + 1,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(2)
        self.current_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.current_label = None
        self.done = False

    def set_transaction(self, transaction):
        # Extract feature keys (all except 'Class') in a sorted order
        feature_keys = sorted([key for key in transaction.keys() if key != 'Class'])
        state = np.array([float(transaction[k]) for k in feature_keys], dtype=np.float32)
        if self.use_pretrained:
            state_tensor = torch.from_numpy(state).unsqueeze(0).double()
            with torch.no_grad():
                pred = self.pretrained_model(state_tensor).item()
            state = np.concatenate([state, np.array([pred], dtype=np.float32)])
        self.current_state = state
        self.current_label = int(transaction['Class'])
        self.done = False
        return self.current_state

    def reset(self):
        self.current_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.current_label = None
        self.done = False
        return self.current_state

    def step(self, action):
        if self.current_label is None:
            raise ValueError("No transaction provided. Call set_transaction() first.")
        if self.current_label == 1:
            reward = 10.0 if action == 1 else -10.0
        else:
            reward = 1.0 if action == 0 else -1.0
        self.done = True
        state = self.current_state.copy()
        info = {"true_label": self.current_label}
        return state, reward, self.done, info

    def render(self, mode='human'):
        print("State:", self.current_state)
        print("True label:", self.current_label)
