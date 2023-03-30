import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import requests


def get_csv():
    training_set_df = pd.read_csv("power_usage.csv")

    training_set = training_set_df.iloc[:, 2:5].values
    training_set = [i for i in training_set if float(i[0]) > -100000]

    return training_set


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    
    return np.array(x), np.array(y)


def scale_data(training_set):
    power_values = np.array([i[0] for i in training_set]).reshape(-1, 1)


if __name__ == "__main__":
    pass