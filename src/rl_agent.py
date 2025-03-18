import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import pandas as pd
import matplotlib.pyplot as plt
from kafka import KafkaConsumer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim=30, output_dim=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).double().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).double().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).double().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).double().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size=30, action_size=2, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        
        self.qnet_local = DQN(input_dim=state_size, output_dim=action_size).double().to(device)
        self.qnet_target = DQN(input_dim=state_size, output_dim=action_size).double().to(device)
        # Use weight decay for L2 regularization
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=5e-4, weight_decay=1e-5)
        
        self.memory = ReplayBuffer(buffer_size=int(1e5), batch_size=64, seed=seed)
        self.t_step = 0
        self.train_loss = []  # For visualizing training loss
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, gamma=0.99)
    
    def epsilon_greedy_action(self, state):
        state_tensor = torch.from_numpy(np.array(state)).double().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state_tensor)
        self.qnet_local.train()
        if random.random() < 0.8:  # 80% chance to exploit
            return int(torch.argmax(action_values).item())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnet_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.train_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet_local.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        self.soft_update(self.qnet_local, self.qnet_target, tau=1e-3)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

def train_agent_kafka(agent, env, n_episodes=100):
    """
    Train the RL agent using Kafka messages as episodes.
    The Kafka consumer is configured to only pick up new messages by using:
      - auto_offset_reset='latest'
      - a unique consumer group id each time.
    After processing each episode, the training loop waits for a keypress.
    """
    unique_group_id = "rl_training_group_" + str(random.randint(1, 1000000))
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',  # Only read new messages
        group_id=unique_group_id,     # Use a unique group id for a fresh offset
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    scores = []
    episode = 0
    print("Starting RL training using Kafka data (waiting for new messages)...")
    for message in consumer:
        transaction = message.value
        print(f"\nReceived transaction for Episode {episode+1}:")
        print(transaction)
        state = env.set_transaction(transaction)
        done = False
        score = 0
        while not done:
            action = agent.epsilon_greedy_action(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
        scores.append(score)
        episode += 1
        print(f"Episode {episode}: Score = {score:.2f}")
        input("Press Enter to proceed to the next episode...")
        if episode >= n_episodes:
            break
    return scores


def main():
    from src.rl_environment import FraudDetectionEnv  # Use absolute import
    env = FraudDetectionEnv(feature_dim=30, pretrained_model_path=None)
    agent = Agent(state_size=30, action_size=2, seed=0)
    
    print("Starting RL training simulation using Kafka data...")
    scores = train_agent_kafka(agent, env, n_episodes=100)
    avg_score = np.mean(scores)
    print("Training complete. Average score:", avg_score)
    
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(scores)+1), scores, marker='o', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("RL Agent Episode Scores (Kafka)")
    plt.grid(True)
    scores_path = os.path.join(model_dir, 'rl_episode_scores_kafka.png')
    plt.savefig(scores_path)
    plt.close()
    print(f"RL episode scores plot saved to {scores_path}")
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(agent.train_loss)+1), agent.train_loss, marker='x', color='blue')
    plt.xlabel("Learning Update Step")
    plt.ylabel("MSE Loss")
    plt.title("RL Agent Training Loss (Kafka)")
    plt.grid(True)
    loss_path = os.path.join(model_dir, 'rl_training_loss_kafka.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"RL training loss plot saved to {loss_path}")
    
    model_path = os.path.join(model_dir, 'rl_model_kafka.pth')
    torch.save(agent.qnet_local.state_dict(), model_path)
    print(f"RL model saved to: {model_path}")

if __name__ == '__main__':
    main()
