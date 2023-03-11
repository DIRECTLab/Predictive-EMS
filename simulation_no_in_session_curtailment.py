import numpy as np
from tqdm import tqdm
class Car:
    def __init__(self, car_name, max_battery_size, max_charge_rate=100_000):
        self.max_charge_rate = max_charge_rate
        self.initial_soc = 0 # Percentage
        self.max_battery_size = max_battery_size # watts
        self.car_name = car_name
        self.charge_percentage = self.initial_soc / self.max_battery_size
    # This will be linear to begin with. Then replace with typical battery curve for a Tesla or something like that. If I can build out a better
    # model, this will be done instead.
    def charge(self):
        pass
    
    def set_initial_soc(self, initial_soc):
        self.initial_soc = initial_soc
    
    def set_max_battery_size(self, max_battery_size):
        self.max_battery_size = max_battery_size
    
    def set_initial_charge_percentage(self, current_soc):
        self.charge_percentage = current_soc * max_battery_size



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


    length_of_time = 10 # min
    current_time = 0
    # Segments will be how many 10 minute increments there are.
    
    peak_consumption = 70000 # Watts
    epochs = 10_000

    # Set up the probabilities of knowing the car that will be charging and predicting the soc
    probability_of_knowing_car = 0.80
    probability_of_predicting_soc = 0.60
    

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




        

        


    
        


