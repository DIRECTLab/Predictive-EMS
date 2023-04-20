class Car:
    def __init__(self, car_name, max_battery_size, max_charge_rate=100_000):
        self.max_charge_rate = max_charge_rate
        self.initial_soc = 0 # watts
        self.max_battery_size = max_battery_size / 1000 # kW
        self.car_name = car_name
        self.soc = self.initial_soc / self.max_battery_size
    
    # This will be linear to begin with. Then replace with typical battery curve for a Tesla or something like that. If I can build out a better
    # model, this will be done instead. Currently linear implementation. This will also be replaced
    def charge(self, charge_amount):
        self.soc += charge_amount / 6 * 0.9 # Assume 90% efficiency of charge rate. Varied amounts due to temperature and currents. Safe to just go with an average of the efficiencies that I saw
    
    def set_max_battery_size(self, max_battery_size):
        self.max_battery_size = max_battery_size
    
    def set_initial_charge_percentage(self, current_soc):
        self.initial_soc = current_soc / 100 * self.max_battery_size
        self.soc = current_soc/ 100 * self.max_battery_size
    
    def reset_charge(self):
        self.soc = self.initial_soc

    # Taking the battery curves, predict when the battery will finish charging. This will be replaced by a prediction algorithm as well.
    def predict_charging_time_completion(self, charging_rate):
        approx_time = self.max_battery_size / (charging_rate * 0.9)
        return approx_time

    def is_charged(self):
        return self.soc > self.max_battery_size
  