import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import namedtuple, deque
import random
import math
from itertools import count
from car import Car
import pandas as pd
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, num_layers):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=state_space, hidden_size=hidden_size * 3, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size * 3, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, action_space)
    
    def forward(self, state):
        state = state.unsqueeze(1)
        h_0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size * 3).to("cuda")
        c_0 = torch.zeros(self.num_layers, state.size(0), self.hidden_size * 3).to("cuda")
        ula, (h_out, _) = self.lstm(state, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size * 3)

        out = F.relu(self.fc1(h_out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))

        out = self.fc5(out)
        return out


class Agent:
    def __init__(self):
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.999
        self.eps = self.eps_start
        self.eps_end = 0.05
        # self.eps_decay = 0.997
        # self.eps_decay = 100
        # self.eps_decay = 1000 # Good for 5000 iterations
        # self.eps_decay = 2000 # Good for 10000 iterations

        self.eps_decay = 500000 # Perfect for 100_000 iterations
        self.tau = 0.005
        self.lr = 1e-4
        self.steps_taken = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = 195 # Let's give 200 actions, each corresponding with a kW curtailment (i.e. 22 is 22000 watts)
        self.state_space = 64

        self.policy_net = DQN(self.state_space, self.action_space, 16, 1).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, 16, 1).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(100_000)
        self.criterion = nn.SmoothL1Loss()
        


    def select_action(self, state):
        sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_taken / self.eps_decay)
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_taken / self.eps_decay)

        self.steps_taken += 1

        if sample > self.eps:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1) # Get rid of the max if we want to do true values. It was doing a lot of negatives and I'm not quite sure the best method to bound it... So maybe this will be the best for rn
        else:
            return torch.tensor([[random.randint(0, 194)]], device=self.device)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
    
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Creates a mask of non-final states (i.e. the car isn't done charging yet)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state) # torch.Size([128, 64])
        action_batch = torch.cat(batch.action) # torch.Size([128, 1])
        reward_batch = torch.cat(batch.reward) # torch.Size([128])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        intermediate = self.policy_net(state_batch) # torch.Size([128, 195])
        state_action_values = intermediate.gather(1, action_batch) # torch.Size([128, 1])


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


def generate_car(correct_car=""):
    # This will load in the configs that I wrote for the electric car dataset and return both the max battery size and the car associated with it

    df = pd.read_csv("car_battery.csv")
    data = df.to_numpy()
    random_index = np.random.randint(len(data))
    car = data[random_index]

    while car[1] == correct_car:
        random_index = np.random.randint(len(data))
        car = data[random_index]
    
    max_battery_size = car[3] * 1000 # Multiplied by 1000 to put into watts
    
    
    # cars = np.array(["Tesla Model S", "Tesla Model 3", "Tesla Model X", "Tesla Model Y"])
    # cars = np.delete(cars, np.where(cars == correct_car))
    # car = np.random.choice(cars)
    # max_battery_size = 100_000 # watts

    return car[1], max_battery_size

def generate_soc():
    distribution = np.random.beta(6, 10)
    soc = round(distribution, 2)
    return soc

def charge(action, car, peak, data, seq_length, rand_location):
    reward = 0
    observation = []
    i = 0
    
    # We don't actually want it to be able to charge at 0 kW, so we add 5000 watts
    charge_rate = (action.cpu().detach().numpy()[0][0] * 1000 + 5000) / 6 # Divide by 6 so you take the charging rate per hour and break it into 10 minute chunks

    while not car.is_charged():
        car.charge(charge_rate)
        reward -= 1

        if charge_rate + data[rand_location + seq_length + i] > peak:
            reward -= 1000
        
        # observation.append(data[rand_location + seq_length + i])
        i += 1
    
    # Let's let the observation be 64 elements (or 640 minutes) to match the size of the state

    for i in range(64):
        observation.append(data[rand_location + seq_length + i])

    car.reset_charge()
    return observation, reward

        
def generate_new_car():
    probability_of_knowing_car = 0.80
    probability_of_predicting_soc = 0.60

    true_car, max_battery_size = generate_car()
    if np.random.random() > probability_of_knowing_car:
        car, max_battery_size = generate_car(true_car)
    else:
        car = true_car

    # Try to determine a good value for max_charging_rate        
    myCar = Car(car, max_battery_size)
    true_soc = generate_soc()

    if np.random.random() > probability_of_predicting_soc:
        soc = generate_soc()
    else:
        soc = true_soc

    myCar.set_initial_charge_percentage(soc)

    return myCar

def rolling_average(rewards):
    rolling_averages = []
    for i in range(len(rewards) - 10):
        rolling_averages.append(np.average(rewards[i:i+10]))

    return rolling_averages

def run():
    
    data = pd.read_csv('power_usage.csv')
    data = data.iloc[:, 2].values
    data = [i for i in data if float(i) > -100_000]
    seq_length = 64
    peak = 70_000

    num_epochs = 10000000
    agent = Agent()
    rewards = []
    actions = []

    for epoch in tqdm(range(num_epochs)):
        myCar = generate_new_car()
        random_start_location = random.randint(0, len(data) - seq_length - 200) # - 100 because most likely, there won't be any charges that take longer than that
        state = data[random_start_location:random_start_location + seq_length]
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        # This is basically like having a while True with a i = 0 and i += 1 inside the loop. Really cool!
        # for t in count():
        action = agent.select_action(state)

        actions.append(action.cpu().detach().numpy()[0][0])
        # TODO: Observation will be the true states during charge, reward will be the accumulated reward. Used to return done, but done is always true here
        observation, reward = charge(action, myCar, peak, data, seq_length, random_start_location)

        rewards.append(reward)
        reward = torch.tensor([reward], device=agent.device)

        # if done:
        #     next_state = None
        # else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
        
        agent.memory.push(state, action, next_state, reward)
        
        # TODO: This will be grabbing a new scenario, not actually changing state to next_state
        # state = next_state
        agent.optimize_model()

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()


        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * agent.tau + target_net_state_dict[key] * (1 - agent.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

    with open("rewards.txt", 'w') as f:
        for line in rewards:
            f.write(str(line) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].plot(rolling_average(rewards))
    ax[0].set_title("Reward over time")
    ax[1].set_ylabel("Reward")
    ax[1].set_xlabel("Epoch")

    ax[1].plot(actions)
    ax[1].set_title("Curtailment over time")
    ax[1].set_ylabel("Curtailment")
    ax[1].set_xlabel("Epoch")
    plt.savefig("rewards_curtailments.jpg")
    plt.show()

if __name__ == "__main__":
    run()


