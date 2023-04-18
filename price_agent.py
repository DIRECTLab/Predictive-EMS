import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class Actor(nn.Module):
    def __init__(self, state_size, action_space):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.action_space)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Critic(nn.Module):
    def __init__(self, state_size, action_space):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_space = action_space
        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out



def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# The way I see this working is as follow:
# 
# Step 1) Upon receiving all the information, this will create new data for each epoch that will take into account the last seq_length of values.
# Step 2) Create a prediction for the curtailment
# Step 3) Simulate charging the car. For every 10 minutes that passes -1 reward, for exceeding peak, -1000
# Step 4) Repeat until convergence
#
# TODO: Add in the agent taking the predictor's input as well

def run(actor, critic, data, num_epochs, car, seq_length, peak):
    optimizer_actor = torch.optim.Adam(actor.parameters())
    optimizer_critic = torch.optim.Adam(critic.parameters())
    rewards = []

    for epoch in tqdm(range(num_epochs)):
        rand_location = np.random.randint(0, len(data))
        reward = 0
        state = data[rand_location:rand_location+seq_length]

        action, value = actor(state), critic(state)

        i = 0
        while not car.is_charged():
            car.charge(action)
            reward -= 1

            if action + data[rand_location + seq_length + i] < peak:
                reward -= 10

            i += 1


        returns = compute_returns()


        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        
        
        optimizer_actor.step()
        optimizer_critic.step()

    torch.save(actor.state_dict(), "models/actor.pth")
    torch.save(critic.save_dict(), "models/critic.pth")



if __name__ == "__main__":
    pass