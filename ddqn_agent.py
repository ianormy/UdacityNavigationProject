"""Implements a Double DQN agent that interacts with and learns from the environment.
"""
import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DdqnAgent:
    """Implements a Double DQN agent that interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99, 
                 tau=1e-3, lr=5e-4, update_every=4):
        """Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): mini-batch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_every (int): how often we learn from our experience
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        # local (learning) model
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # target model
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # use an Adam optimizer with our local model
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize the time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """This method is called each training step with our (s,a,r,s',done)
        experience tuple.
        """
        # Save the experience tuple in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in replay memory then get a random subset and learn from it
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns an action for a given state using the current policy.
        
        Args:
            state (array_like): current state
            eps (float): epsilon. Used for epsilon-greedy action selection
        Returns:
            action (int): action to take
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experience, gamma):
        """Update value parameters using the provided experience tuple.

        Args:
            experience (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experience
        # Get the max predicted Q values (for next states) from the local model
        argmax_a_q_sp = self.qnetwork_local(next_states).detach().max(1)[1]
        q_sp = self.qnetwork_target(next_states).detach()
        q_targets_next = q_sp[np.arange(self.batch_size), argmax_a_q_sp].unsqueeze(1)
        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        # back propagate the loss through the local (learning) model
        loss.backward()
        self.optimizer.step()
        # update the target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
