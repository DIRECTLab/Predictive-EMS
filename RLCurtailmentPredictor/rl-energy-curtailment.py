import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque, namedtuple
import random
import math
from car import Car
import pandas as pd
import matplotlib.pyplot as plt
from Transformer import *

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
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()

        
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_space)
    
    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = self.fc4(out)

        return out

class Agent:
    def __init__(self, state_space, action_space=195):
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1
        self.epsilon = self.eps_start
        self.min_epsilon = 0.05

        self.epsilon_decay = 0.99985
        self.tau = 0.005
        self.lr = 1e-4
        self.losses = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_space = action_space # Give actions in 10s of kW. (So 1 is 10kw)
        self.state_space = state_space

        self.policy_net = DQN(self.state_space, self.action_space).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(100_000)
        self.criterion = nn.SmoothL1Loss().to(self.device)


    def decay(self):
        self.epsilon *= self.epsilon_decay

    def act(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[random.randint(0, self.action_space - 1)]], device=self.device)
        with torch.no_grad():
            # action = np.argmax(self.policy_net(state).detach().cpu())
            return self.policy_net(state).max(1)[1].view(1, 1)


    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        random.seed(1)
        
        samples = self.memory.sample(self.batch_size)
        
        # Compute Q(s_t)
        batch = Transition(*zip(*samples))


        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action) # torch.Size([24, 1])
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        intermediate = self.policy_net(state_batch) #torch.Size([24, 19])
        state_action_values = intermediate.gather(1, action_batch)

        
        # Compute Q(s_t+1) using the older network
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0] # Predict on the next state
        
        expected_state_action_values = next_state_values * self.gamma + reward_batch

        # for i, value in enumerate(next_state_values):
        #     expected_state_action_values.append(value * self.gamma + rewards[i])

        # expected_state_action_values = (next_state_values * self.gamma) + np.array(rewards)

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss.detach().cpu().item())
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.decay()
        self.lastSample = samples



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
    # charge_rate = (action.cpu().detach().numpy()[0][0] * 1000 + 5000) / 6 # Divide by 6 so you take the charging rate per hour and break it into 10 minute chunks
    charge_rate = (action * 1000 + 10000) / 6

    while not car.is_charged():
        car.charge(charge_rate)
        reward -= 1

        if charge_rate + data[rand_location + seq_length + i] > peak:
            reward -= 100
        
        # observation.append(data[rand_location + seq_length + i])
        i += 1
    
    # Let's let the observation be 64 elements (or 640 minutes) to match the size of the state

    for i in range(seq_length):
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

def generate_energy_usage():
    usage = pd.read_csv('power_usage.csv')

    power = usage.iloc[:, 2].to_numpy()
    ten_minute_averages = []
    
    for i in power:
        ten_minute_averages.append(int(math.ceil(i / 1000 + 100)))

    return np.clip(ten_minute_averages, 0, 300_000 / 1000 + 3)

def rolling_average(rewards):
    rolling_averages = []
    for i in range(len(rewards) - 10):
        rolling_averages.append(np.average(rewards[i:i+10]))

    return rolling_averages


if __name__ == "__main__":
    data = generate_energy_usage()
    seq_length = 40
    # peak = 70

    num_epochs = 20000
    agent = Agent(seq_length+1)
    rewards = []
    actions = []

    num_head = 16
    num_encoder_layer = 8
    num_decoder_layer = 8
    length_of_prediction = 18
    num_tokens = int(300_000 / 1000 + 3)

    energy_model = Transformer(num_tokens=num_tokens, dim_model=256, num_heads=num_head, num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer, dropout=0.2)
    energy_model.load_state_dict(torch.load('transformer_energy_predictor.pth'))
    energy_model.to(agent.device)
    peak = random.randint(50, 140)
    print(f"Peak Load: {peak}kW")

    for epoch in tqdm(range(num_epochs)):
        myCar = generate_new_car()
        random_start_location = random.randint(0, len(data) - seq_length - 10) # This will be fed into the energy predictor, so only needs 40 sequence values
        
        
        state = transformer_predict(energy_model, torch.tensor(np.array([data[random_start_location:random_start_location+seq_length]]), dtype=torch.long, device=agent.device), device=agent.device)
        state = state[1:-1]
        state = state[:seq_length]
        state.append(peak)

        state = torch.tensor(state, device=agent.device, dtype=torch.float32).unsqueeze(0)
        
        action = agent.act(state)
        actions.append(action.detach().cpu().numpy()[0][0] + 10)
        # print(f"{action.detach().cpu().numpy()[0][0] + 10} kW")

        observation, reward = charge(action, myCar, peak, data, seq_length, random_start_location)
        rewards.append(reward)
        observation.append(peak)
        next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
        reward = torch.tensor([reward], device=agent.device)

        agent.memory.push(state, action, next_state, reward)

        agent.learn()

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
    
    with open("losses.txt", 'w') as f:
        for loss in agent.losses:
            f.write(str(line) + "\n")
    
    with open('actions.txt', "w") as f:
        for loss in actions:
            f.write(str(line) + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    ax[0].plot(rolling_average(rewards))
    ax[0].set_title("Reward over time")
    ax[0].set_ylabel("Reward")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(actions)
    ax[1].set_title("Curtailment over time")
    ax[1].set_ylabel("Curtailment")
    ax[1].set_xlabel("Epoch")


    ax[2].plot(agent.losses)
    ax[2].set_title("Loss Over Epochs")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlabel("Epoch")

    plt.savefig("rewards_curtailments.jpg")
    plt.show()
    print(agent.epsilon)
    torch.save(agent.policy_net.state_dict(), "curtailment_agent.pth")


