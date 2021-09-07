import random
import numpy as np
import torch


# memory replay
class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.idx = 0

    def store(self, state, action, next_state, reward):

        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward

        self.idx = (self.idx+1)%self.capacity # for circular memory

    def sample(self, batch_size, device):
        
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)

        return states, actions, next_states, rewards
        

    def __len__(self):
        return len(self.buffer_state)
    
    
    
    
