import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length=5):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size * 3, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 1)
        self.fc2 = nn.Linear(hidden_size * 1, num_classes)
        # self.fc3 = nn.Linear(hidden_size * 1, hidden_size // 2)
        # self.fc4 = nn.Linear(hidden_size // 2, num_classes)



    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size * 3)).to("cuda")
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size * 3)).to("cuda")
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size * 3)
        out = self.fc(h_out)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc4(out)

        return out


    # def sliding_windows(self, data):
    #     x = []
    #     y = []

    #     for i in range(len(data) - self.seq_length - 1):
    #         _x = data[i:(i+self.seq_length)]
    #         _y = data[i+self.seq_length]
    #         x.append(_x)
    #         y.append(_y)
        
    #     return np.array(x), np.array(y)
    






        

    