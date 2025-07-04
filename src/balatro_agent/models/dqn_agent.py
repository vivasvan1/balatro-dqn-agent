import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        # Larger network for more complex Balatro state space
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Define the Replay Buffer
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        lr=5e-4,
        tau=1e-3,
        update_every=4,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.01,
        seed=0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_every = update_every
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).float()
        self.qnetwork_target = QNetwork(state_size, action_size).float()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.training_loss = []
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def get_action(self, state, eps=None):
        """Returns actions for given state as per current policy."""
        if eps is None:
            eps = self.epsilon
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.training_loss.append(loss.item())
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def decay_epsilon(self):
        """Decrease epsilon."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """Save the weights of the local Q-network."""
        torch.save(self.qnetwork_local.state_dict(), filename+'_local.pth')
        torch.save(self.qnetwork_target.state_dict(), filename+'_target.pth')
    
    def load(self, filename):
        """Load weights into the local Q-network."""
        self.qnetwork_local.load_state_dict(torch.load(filename+"_local.pth", weights_only=True))
        self.qnetwork_target.load_state_dict(torch.load(filename+"_target.pth", weights_only=True))
        self.qnetwork_local.eval()
        self.qnetwork_target.eval() 