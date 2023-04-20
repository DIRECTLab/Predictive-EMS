import matplotlib.pyplot as plt
import requests
import numpy as np
import datetime
import pandas as pd
from fitter import Fitter, get_common_distributions
import scipy


class BayesUpdater:
    def __init__(self):
        self.df = pd.read_csv("Simulation/charge_information.csv", usecols=(1, 2))

        # Grab percent_full column
        self.percentage_charged = self.df.iloc[:, 1].to_numpy()
        self.bin_labels()

    def determine_distribution(self, data, title=""):
        f = Fitter(data, distributions=get_common_distributions())
        f.fit()
        f.summary()
        plt.title(title)
        plt.xlabel("Battery SOC")
        plt.ylabel("Occurences")
        return f.get_best()

    def bin_labels(self):
        values = np.zeros(20)
        labels = [5 * i for i in range(20)]

        for charge in self.percentage_charged:
            if charge < 0.95:
                charge = int(round(charge, 2) * 100)
                index = charge // 5
                values[index] += 1
        
        times_mostly_charged = 0
        times_partially_charged = 0
        times_barely_charged = 0
        mostly_charged = []
        partially_charged = []
        barely_charged = []

        total_count = len(self.percentage_charged)

        for charge in self.percentage_charged:
            # These don't make sense, other than they were caused during testing and starting and stopping the charge manually. This isn't an accurate amount of charge
            if charge > 0.95: 
                continue 
            
            if charge < 0.5:
                times_barely_charged += 1
                barely_charged.append(charge)
            elif charge < 0.75:
                times_partially_charged += 1
                partially_charged.append(charge)
            else:
                times_mostly_charged += 1
                mostly_charged.append(charge)

        # print(f"Times mostly charge: {times_mostly_charged}/{total_count}")
        # print(f"Times partially charge: {times_partially_charged}/{total_count}")
        # print(f"Times barely charge: {times_barely_charged}/{total_count}")

        self.likelihood_mostly_charged = times_mostly_charged / total_count
        self.likelihood_partially_charged = times_partially_charged / total_count
        self.likelihood_barely_charged = times_barely_charged / total_count

        # likelihood_mostly_charged = likelihood(pi, times_mostly_charged, total_count)
        # likelihood_partially_charged = likelihood(pi, times_partially_charged, total_count)
        # likelihood_barely_charged = likelihood(pi, times_barely_charged, total_count)

        # print(f"Mostly Charged: {round((times_mostly_charged / total_count) * 100, 2)}%")
        # print(f"Partially Charged: {round((self.likelihood_partially_charged) * 100, 2)}%")
        # print(f"Barely Charged: {round((self.likelihood_barely_charged) * 100, 2)}%")

        self.mostly_charged_distribution = self.determine_distribution(mostly_charged, "Mostly Charged")
        self.partially_charged_distribution = self.determine_distribution(partially_charged, "Partially Charged")
        self.barely_charged_distribution = self.determine_distribution(barely_charged, "Barely Charged")

    
    # {'powerlaw': {'a': 1.7499123157860441, 'loc': 0.739214691236262, 'scale': 0.20836425616587315}}
    def generate_random_variable_power_law(self, data):
        r = scipy.stats.powerlaw.rvs(a=data['a'], loc=data['loc'], scale=data['scale'], size=1)
        return r

    # {'uniform': {'loc': 0.5040736842105263, 'scale': 0.24376842105263152}}
    def generate_random_variable_uniform(self, data):
        r = scipy.stats.uniform.rvs(loc=data['loc'], scale=data['scale'], size=1)
        return r

    # {'cauchy': {'loc': 0.4046345646053891, 'scale': 0.025979654757393673}}
    def generate_random_variable_cauchy(self, data):
        r = scipy.stats.cauchy.rvs(loc=data['loc'], scale=data['scale'], size=1)
        r = np.clip(r, 0.1, 0.5)

        return r
    

    def generate_battery_value(self):
        random_number = np.random.uniform(0, 1)
        if random_number < self.likelihood_barely_charged:
            val = int(round(self.generate_random_variable_cauchy(self.barely_charged_distribution['cauchy'])[0], 2) * 100)
        elif random_number < self.likelihood_barely_charged + self.likelihood_partially_charged:
            val = int(round(self.generate_random_variable_uniform(self.partially_charged_distribution['uniform'])[0], 2) * 100)
        else:
            val = int(round(self.generate_random_variable_power_law(self.mostly_charged_distribution['powerlaw'])[0], 2) * 100)

        return val
