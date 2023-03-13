import numpy as np
from tqdm import tqdm

class Car:
    def __init__(self, car_name, max_battery_size, max_charge_rate=100_000):
        self.max_charge_rate = max_charge_rate
        self.initial_soc = 0 # watts
        self.max_battery_size = max_battery_size # watts
        self.car_name = car_name
        self.soc = self.initial_soc / self.max_battery_size
    
    # This will be linear to begin with. Then replace with typical battery curve for a Tesla or something like that. If I can build out a better
    # model, this will be done instead. Currently linear implementation. This will also be replaced
    def charge(self, charge_amount):
        self.soc += charge_amount
    
    def set_max_battery_size(self, max_battery_size):
        self.max_battery_size = max_battery_size
    
    def set_initial_charge_percentage(self, current_soc):
        self.initial_soc = current_soc * max_battery_size
        self.soc = current_soc * max_battery_size
    
    def reset_charge(self):
        self.soc = self.initial_soc

    # Taking the battery curves, predict when the battery will finish charging. This will be replaced by a prediction algorithm as well.
    def predict_charging_time_completion(self, charging_rate):
        pass


        


# This will be the thing that generates the "true" energy usage. Will eventually be based on historical data instead.
def generate_energy_usage():
    usage = np.random.normal(30000, 15000, size=1000)

    return usage

# This will eventually be replaced by a LSTM prediction engine instead
def predict_energy_usage(current_usage):
    usage = np.random.normal(current_usage, 15000, size=1000)

    return usage


# This will return an optimal charging rate. Will eventually be replaced by a RL algorithm instead. Right now, it will just try incrementing the power output by 1000 watts each time and if it can finish before it predicts it will exceed the peak, return that value (well, keep going until it does exceed the peak)
def determine_optimal_charging_rate(myCar, predicted_energy_usage, peak):
    best_charge_rate = 0
    current_charge = 0

    while current_charge + 5000 / 6 < myCar.max_charge_rate:
        exceeded_peak = False

        current_charge += 5000 / 6
        current_charge = round(current_charge)
        i = 0
        while myCar.soc < myCar.max_battery_size:
            myCar.charge(current_charge)

            if predicted_energy_usage[i] + current_charge > peak:
                exceeded_peak = True
                break

            i += 1
        myCar.reset_charge()

        if not exceeded_peak:
            best_charge_rate = current_charge

    return best_charge_rate


def generate_car(correct_car=""):
    # This will load in the configs that I wrote for the electric car dataset and return both the max battery size and the car associated with it

    cars = np.array(["Tesla Model S", "Tesla Model 3", "Tesla Model X", "Tesla Model Y"])
    
    cars = np.delete(cars, np.where(cars == correct_car))
    car = np.random.choice(cars)
    max_battery_size = 100_000 # watts

    return car, max_battery_size

def generate_soc():
    distribution = np.random.beta(6, 10)
    soc = round(distribution, 2)
    return soc



if __name__ == "__main__":
    peak_consumption = 70000 # Watts
    epochs = 10_000

    # Set up the probabilities of knowing the car that will be charging and predicting the soc
    probability_of_knowing_car = 0.80
    probability_of_predicting_soc = 0.60
    charge_rates = []
    

    # This simulation will "spawn" a car, and it is our job to charge it. We will have a basic probability distribution that will generate a power consumption for the next hour or so
    # Based upon this generation, and the chance of us knowing what soc they are at and what car they have (i.e. battery size), we will then set a curtailment to charge them as quick as possible but without
    # exceeding the peak. Exceeding the peak is very bad. This first simulation will not be able to curtail in session, which will make it more interesting. The balance between fast charging and 
    # cost saving will be the reward. 
    # Potential Reward Structure
    # -1 reward for every 10 minutes it isn't done charging, -1000 for exceeding peak_consumption.
    for _ in tqdm(range(epochs)):
        # Generate which car came up to drive
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

        true_energy_usage = generate_energy_usage()
        predicted_energy_usage = predict_energy_usage(true_energy_usage[0])

        
        charge_rate = determine_optimal_charging_rate(myCar, predicted_energy_usage, peak_consumption) # Since it is every 10 minutes, and it is 10_000 watts per hour, you need to do 1/6 of the amount
        
        # Now that you got your charge_rate, you then actually charge the car. However, this SOC may be different than the one that you tried determining it on.


    print(np.average(charge_rates))