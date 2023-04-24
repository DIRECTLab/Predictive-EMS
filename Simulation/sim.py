import numpy as np
from tqdm import tqdm
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
# from sklearn.preprocessing import MinMaxScaler
from car import Car
from Transformer import *
from agent import Agent
from bayesian_updater import BayesUpdater
import random

# This will be the thing that generates the "true" energy usage. Will eventually be based on historical data instead. (Get the past 4 days at the leviton)
def generate_energy_usage():
    today = datetime.datetime.now()

    date_end = datetime.date.today()

    delta_end = datetime.timedelta(days=1)
    delta_start = datetime.timedelta(days=4)
    date_end = date_end - delta_end
    date_start = today - delta_start

    date_end = date_end.strftime("%Y-%m-%d")
    date_start = date_start.strftime("%Y-%m-%d")
    print(date_start)

    print(date_end)

    response = requests.get(f"http://144.39.204.242:11236/evr/leviton/evr?dateStart={date_start}&dateEnd={date_end}")
    usage = response.json()
    ten_minute_averages = []

    # Chunk it into 10 minute increments. Change this to 120 in a week or so
    for i in range(0, len(usage['data']), 85):
        values = usage['data'][i:i+120]
        power = [value['power'] for value in values]

        ten_minute_averages.append((values[0]['timestamp'], int(np.clip(math.ceil((np.average(power) * 1000) / 1000) + 100, 0, 300_000 / 1000 + 3))))

    
    power_usage = np.array([i[1] for i in ten_minute_averages])
    np.savetxt("energy_usage_four_days.csv", power_usage, delimiter=",")

    return ten_minute_averages

def use_energy_usage():
    return pd.read_csv("energy_usage_four_days.csv").to_numpy()

def use_all_energy_usage():
    df = pd.read_csv("Simulation/power_usage.csv", usecols=["power"])

    
    power = df.to_numpy().flatten()

    for i in range(len(power)):
        power[i] = int(np.clip(math.ceil(power[i] / 1000 + 100), 0,  300_000 / 1000 + 3))
    
    # for i in range(len(power)):
    #     power[i] = int(math.ceil(i * 1000) / 1000 + 100)
    #     if power[i] > 300_000 / 1000 + 3:
    #         power[i] = 300_000 / 1000 + 3
    #     elif power[i] < 0:
    #         power[i] = 0
    
    return power




def sliding_windows(data, lstm):
    x = []
    y = []

    for i in range(len(data) - lstm.seq_length - 1):
        _x = data[i:(i+lstm.seq_length)]
        _y = data[i+lstm.seq_length]
        x.append(_x)
        y.append(_y)
    
    return np.array(x), np.array(y)

def generate_car(correct_car=""):
    # This will load in the configs that I wrote for the electric car dataset and return both the max battery size and the car associated with it

    df = pd.read_csv("Simulation/car_battery.csv")
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




def generate_new_car(bayes_updater):
    probability_of_knowing_car = 0.80

    true_car, max_battery_size = generate_car()
    if np.random.random() > probability_of_knowing_car:
        car, max_battery_size = generate_car(true_car)
    else:
        car = true_car

    # Try to determine a good value for max_charging_rate        
    myCar = Car(car, max_battery_size)
    soc = bayes_updater.generate_battery_value()


    myCar.set_initial_charge_percentage(soc)

    return myCar



def setup_energy_prediction_agent(device):
    num_head = 16
    num_encoder_layer = 8
    num_decoder_layer = 8
    num_tokens = int(300_000 / 1000 + 3)

    energy_model = Transformer(num_tokens=num_tokens, dim_model=256, num_heads=num_head, num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer, dropout=0.2)
    energy_model.load_state_dict(torch.load('models/transformer_energy_predictor.pth'))
    energy_model.to(device)

    return energy_model


if __name__ == "__main__":
    peak = 100 #kW
    # power_usage = generate_energy_usage()
    # power_usage = np.array([i[1] for i in power_usage])
    # power_usage = use_energy_usage().flatten()
    power_usage = use_all_energy_usage()
    probability_of_knowing_car = 0.80
    length_of_prediction = 18
    epochs = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_length = 40
    state_space = seq_length + 1
    action_space = 195

    static_curtailments = [50, 100, 150, 200, 250]

    # Setup energy model
    energy_model = setup_energy_prediction_agent(device)

    # Setup Curtailment Model
    rl_agent = Agent("models/curtailment_agent.pth", device, state_space, action_space)

    # Setup Bayesian Battery Predictor
    bayes_updater = BayesUpdater()

    charge_rates = []
    static_curtailment_peak_exceed = [0 for _ in range(len(static_curtailments))]
    static_curtailment_time = [0 for _ in range(len(static_curtailments))]
    rl_peak_exceed = 0
    rl_curtailment_time = 0


    for epoch in tqdm(range(epochs)):
        myCar = generate_new_car(bayes_updater)
        random_start_location = random.randint(0, len(power_usage) - seq_length - 10) # This will be fed into the energy predictor, so only needs 40 sequence values

        predicted_energy_usage = transformer_predict(energy_model, torch.tensor(np.array([power_usage[random_start_location:random_start_location+seq_length]]), dtype=torch.long, device=device), device=device)
        predicted_energy_usage = predicted_energy_usage[1:-1]
        predicted_energy_usage = predicted_energy_usage[:40]
        while len(predicted_energy_usage) < 40:
            predicted_energy_usage.append(predicted_energy_usage[-1])
        predicted_energy_usage.append(peak)
        charge_rate = rl_agent.predict(predicted_energy_usage, device)[0][0]

        charge_rates.append(charge_rate)

        for i in range(len(static_curtailments) + 1):
            j = 0
            while not myCar.is_charged():
                if i < len(static_curtailments):
                    myCar.charge(static_curtailments[i])
                    if static_curtailments[i] + power_usage[j] - 100 > peak:
                        static_curtailment_peak_exceed[i] += 1
                    static_curtailment_time[i] += 1

                else:
                    while not myCar.is_charged():
                        myCar.charge(charge_rate)
                        if charge_rate + power_usage[j] - 100 > peak:
                            rl_peak_exceed += 1
                        rl_curtailment_time += 1
                j += 1
            myCar.reset_charge()

    for i in range(len(static_curtailments)):
        print(f"Average time to completion for {static_curtailments[i]} kW was {(static_curtailment_time[i]/epochs) * 10} minutes")
        print(f"Average times exceeding peak for {static_curtailments[i]} kW was {static_curtailment_peak_exceed[i]/epochs}")
        print()

    print(f"Average time to completion for RL agent was {(rl_curtailment_time/epochs) * 10} minutes")
    print(f"Average times exceeding peak for RL agent was {rl_peak_exceed/epochs}")
    print(f"RL Agent had an average curtailment of {np.average(charge_rates)}")



