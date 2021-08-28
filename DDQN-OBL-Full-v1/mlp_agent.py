import numpy as np
import gin.tf
from rl_env import Agent
from third_party.dopamine import sum_tree as dopamine_sum_tree
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import time
from itertools import count
import math
import csv
import pickle
import random
import os
from collections import defaultdict

DEFAULT_PRIORITY = 100
MAX_SAMPLE_ATTEMPTS = 1000000

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus

@gin.configurable
class MLP(nn.Module):
    def __init__(self, n_actions, observation_size, hidden_size=512):
        super(MLP, self).__init__()
        self.n_actions = n_actions
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        # Create fully connected layers
        self.fc1 = nn.Linear(self.observation_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.n_actions)
    
    def forward(self, observation):
        # Build the MLP for a forward pass
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q_values = self.fc3(x)
        return q_values
    
    def act(self, observation, legal_actions, epsilon):
        q_values_tensor = self.forward(observation)
        q_values_np = q_values_tensor[0].cpu().detach().numpy()
        # Create list with q_values for legal actions only
        q_values = []
        legal_action_indices = np.where(legal_actions == 0.0)[0]
        for idx in range(len(legal_actions)):
          if idx in legal_action_indices:
            q_values.append(q_values_np[idx])
          else:
            q_values.append(-np.Inf)
        # Take epsilon-greedy action
        if np.random.uniform() > epsilon:
          action = np.nanargmax(q_values)
        else:
          action = np.random.choice(legal_action_indices)
        return action

@gin.configurable
class ExpBuffer():
    def __init__(self, max_storage):
        self.max_storage = max_storage
        self.counter = -1
        self.storage = [0 for i in range(self.max_storage)]
        self.sum_tree = dopamine_sum_tree.SumTree(self.max_storage)
        for index in range(self.max_storage):
            self.sum_tree.set(index, 0.0)
        self.batch_indices = []

    def write_tuple(self, oarodl, priority = DEFAULT_PRIORITY, index = None):
        if self.counter < self.max_storage-1:
            self.counter +=1
        else:
            self.counter = 0
        self.storage[self.counter] = oarodl
        if index is None:
            index = self.counter

        self.sum_tree.set(index, priority)

    def set_priority(self, indices, priorities):
        for i, memory_index in enumerate(indices):
            self.sum_tree.set(memory_index, priorities[i])

    def get_priority(self, indices):
        priority_batch = np.empty((len(indices)), dtype = np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return torch.tensor(priority_batch, dtype = torch.float32).cuda()
    
    def sample(self, batch_size):
        # Returns sizes of (batch_size, *) depending on action/observation/return/done
        self.batch_indices = []
        allowed_attempts = MAX_SAMPLE_ATTEMPTS
        while len(self.batch_indices) < batch_size and allowed_attempts > 0:
            try:
                index = self.sum_tree.sample()
                if self.storage[index] != 0:
                    self.batch_indices.append(index)
                else:
                    allowed_attempts -= 1
            except:
                print("index: ", index)
                print(self.sum_tree.nodes[-1])
                raise Exception
        samples = np.empty((batch_size), dtype = object)
        for i, memory_index in enumerate(self.batch_indices):
            samples[i] = self.storage[memory_index]
        last_observations, actions, rewards, observations, dones, legal_actions = zip(*samples)
        return torch.tensor(last_observations, dtype = torch.float32).cuda(),\
               torch.tensor(actions).cuda(),\
               torch.tensor(rewards).float().cuda(),\
               torch.tensor(observations, dtype = torch.float32).cuda(),\
               torch.tensor(dones).cuda(),\
               torch.tensor(legal_actions, dtype = torch.float32).cuda()

@gin.configurable
class MLPAgent(Agent):

  @gin.configurable
  def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               replay_buffer_size=100000,
               batch_size = 32,
               epsilon_train = 0.02,
               epsilon_eval = 0.001,
               epsilon_decay_period = 1000,
               gamma = 0.99,
               learning_rate = 0.000025,
               optimizer_epsilon = 0.00003125,
               explore = 500,
               clip_grad_norm = 0.1,
               weight_decay = 0.95,
               last_action = {},
               last_observation = {},
               begin = defaultdict(lambda: 1),
               i_episode = 0,
               training_steps = 0,
               update_period = 4,
               target_update_period = 500,
               loss = 0,
               epsilon_fn=linearly_decaying_epsilon,
               tf_device='/cpu:*',):
    self.num_actions = num_actions
    self.observation_size = observation_size
    self.num_players = num_players
    self.replay_buffer_size = replay_buffer_size
    self.batch_size = batch_size
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.optimizer_epsilon = optimizer_epsilon
    self.explore = explore
    self.i_episode = i_episode
    self.training_steps = training_steps
    self.update_period = update_period
    self.target_update_period = target_update_period
    self.loss = loss
    self.epsilon_fn = epsilon_fn
    self.eval_mode = False
    self.raw_data = [["training_step", "example_reward", "example_done", "avg_q_value", "avg_target_value", 
                  "loss", "fc1_weight_NORM", "fc1_bias_NORM", "fc2_weight_NORM", "fc2_bias_NORM",
                  "fc3_weight_NORM", "fc3_bias_NORM"]]
    # ExpBuffer
    self.replay_buffer = ExpBuffer(self.replay_buffer_size)
    # MLP
    self.mlp = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target = MLP(self.num_actions, self.observation_size).cuda()
    self.mlp_target.load_state_dict(self.mlp.state_dict())

    # Optimizer
    self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr = self.learning_rate, eps = self.optimizer_epsilon)
    
  def write_fictitious_tuple(self, aorodl):
    
    aorodl_copy = copy.deepcopy(aorodl)
    self.replay_buffer.write_tuple(aorodl_copy)

  def step(self, legal_actions, observation):
    if self.i_episode > self.explore:
      self.loss = self._update_network()
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)
    action = self.mlp.act(
      torch.tensor(observation).float().view(1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)
    return action, self.loss

  def fictitious_step(self, legal_actions, observation):
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.explore, self.epsilon_train)
    action = self.mlp.act(
      torch.tensor(observation).float().view(1,-1).cuda(),
      legal_actions,
      epsilon = epsilon)
    return action

  def end_episode(self):
    self.i_episode += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    return None

  def _update_network(self):

    if self.eval_mode:
      return 0
    if self.training_steps % self.update_period == 0:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                 self.explore, self.epsilon_train)
      last_observations, actions, rewards, observations, dones, legal_actions = self.replay_buffer.sample(self.batch_size)
      # Get indices of the experiences (indices correspond to the buffer and the sum_tree object)
      batch_indices = copy.deepcopy(self.replay_buffer.batch_indices)

      # Get priorities of each experience in batch
      target_priorities = self.replay_buffer.get_priority(batch_indices)

      # Convert priorities to importance-sampling weight
      target_priorities = torch.add(target_priorities, 1e-10)
      target_priorities = 1.0 / torch.sqrt(target_priorities)
      target_priorities /= torch.max(target_priorities, -1)[0]

      # Run a forward pass to get Q-values from last observation
      q_values = self.mlp.forward(last_observations).float()

      # Get q-value corresponding to the last action
      q_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)

      # Apply online network to observations to get corresponding q-values
      pred_q_values_online = self.mlp.forward(observations).float()

      # Convert all zeroes to ones in legal actions
      legal_actions[legal_actions == 0] = 1

      # Multiply q values to legal actions to get q values for legal actions only
      pred_q_values_legal = pred_q_values_online*legal_actions

      # Negative q values multiplied by illegal actions (-Inf) result in +Inf. Change it back to -Inf.
      pred_q_values_legal[pred_q_values_legal == np.Inf] = -np.Inf

      # Get the index corresponding to the max q-value
      action_pred = torch.max(pred_q_values_legal, dim = -1)[1] # debug
      max_q_value = torch.max(pred_q_values_legal, dim = -1)[0] # debug
      # Now apply target network to observations
      pred_q_values = self.mlp_target.forward(observations).float()

      # Get q-value corresponing to action_pred
      pred_q_value = torch.gather(pred_q_values, -1, action_pred.unsqueeze(-1)).squeeze(-1)

      # Compute target values
      target_values = rewards + (self.gamma * (1 - dones.float()) * pred_q_value)

      #Update network parameters
      self.optimizer.zero_grad()
      # Get individual losses related to each experience in the batch
      losses = nn.MSELoss(reduction = 'none')(q_values , target_values.detach())

      # Calculate priorities as sqrt of losses (plus a tiny epsilon)
      priorities = torch.sqrt(torch.add(losses, 1e-10))

      # Set the new priorities for the experiences in the buffer
      self.replay_buffer.set_priority(batch_indices, priorities)
      
      # Do backprop based on weighted loss
      weighted_losses = target_priorities * losses

      weighted_loss = torch.mean(weighted_losses)

      weighted_loss.backward()
    #   nn.utils.clip_grad_norm_(self.mlp.parameters(), self.clip_grad_norm) # debug
      self.optimizer.step()

      self.loss = weighted_loss.item()
      fc1_weight, fc1_bias, fc2_weight, fc2_bias = None, None, None, None
      for name, param in self.mlp.named_parameters():
        if name == "fc1.weight":
          fc1_weight = param.grad.norm().item()
        if name == "fc1.bias":
          fc1_bias = param.grad.norm().item()
        if name == "fc2.weight":
          fc2_weight = param.grad.norm().item()
        if name == "fc2.bias":
          fc2_bias = param.grad.norm().item()
        if name == "fc3.weight":
          fc3_weight = param.grad.norm().item()
        if name == "fc3.bias":
          fc3_bias = param.grad.norm().item()
    if self.training_steps % self.target_update_period == 0:
      lst = [self.training_steps, rewards[0].item(), dones[0].item(),
             torch.mean(q_values, 0).item(),torch.mean(target_values, 0).item(),
             self.loss, fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias]
      self.raw_data.append(lst)
      with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(self.raw_data)
      self.mlp_target.load_state_dict(self.mlp.state_dict())
    self.training_steps += 1
    return self.loss
