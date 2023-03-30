import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import requests
import datetime

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



    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size * 3)).to("cuda")
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size * 3)).to("cuda")

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size * 3)
        out = self.fc(h_out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out



def get_csv():
    training_set_df = pd.read_csv("power_usage.csv")

    training_set = training_set_df.iloc[:, 2:5].values
    training_set = [i for i in training_set if float(i[0]) > -100000]

    return training_set



def sliding_windows(data, seq_length):
    x = []
    y = []

    prediction_length = 8


    for i in range(len(data) - seq_length - prediction_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:i+seq_length+prediction_length]
        x.append(_x)
        y.append(_y)
    
    return np.array(x), np.array(y)


def train_best_model(learning_rates, seq_lengths, hidden_sizes, num_classes, input_size, num_layers, sc, training_data):
    for lr in learning_rates:
        for _, seq_length in enumerate(seq_lengths):

            x, y = sliding_windows(training_data, seq_length)
            train_size = int(len(y) * 0.67)
            test_size = len(y) - train_size

            dataX = torch.Tensor(np.array(x))
            dataY = torch.Tensor(np.array(y))

            trainX = torch.Tensor(np.array(x[0:train_size]))
            trainY = torch.Tensor(np.array(y[0:train_size]))

            testX = torch.Tensor(np.array(x[train_size:len(x)]))
            testY = torch.Tensor(np.array(y[train_size:len(y)]))


            for _, hidden_size in enumerate(hidden_sizes):
                print(f"Starting with a sequence length of {seq_length} and hidden_size of {hidden_size}")
                lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
                lstm = lstm.to(device)
                early_stop = 50
                least_loss = np.inf
                criterion = torch.nn.MSELoss()    
                optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

                # Train the model
                for epoch in range(num_epochs):
                    trainX = trainX.to(device)
                    trainY = trainY.to(device)
                    outputs = lstm(trainX)
                    optimizer.zero_grad()
                    
                    # obtain the loss function
                    loss = criterion(outputs, trainY)
                    loss.backward()
                    optimizer.step()
                    
                    if least_loss > loss.item():
                        least_loss = loss.item()
                        early_stop = 50
                    else:
                        early_stop -= 1

                    if epoch % 100 == 0:
                        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
                    
                    if early_stop <= 0:
                        print("Epoch: %d, loss: %1.5f, best_loss: %1.5f" % (epoch, loss.item(), least_loss))
                        break

                lstm.eval()
                dataX = dataX.to(device)
                dataY = dataY.to(device)
                train_predict = lstm(dataX)

                data_predict = train_predict.cpu().data.numpy()
                dataY_plot = dataY.cpu().data.numpy()

                data_predict = sc.inverse_transform(data_predict)
                dataY_plot = sc.inverse_transform(dataY_plot)


                lstm.eval()
                testX = testX.to(device)
                testY = testY.to(device)
                test_predictions = lstm(testX)
                test_loss = criterion(test_predictions, testY)

                if test_loss.item() < lowest_loss:
                    lowest_loss = test_loss
                    best_seq = seq_length
                    best_hidden_size = hidden_size
                    best_lr = lr

        print(f"Best Loss: {lowest_loss} on sequence length {best_seq}, hidden size of {best_hidden_size} and learning rate of {best_lr}")
        return lowest_loss, best_seq, best_hidden_size, best_lr


def train_single_best(best_seq, best_hidden_size, best_lr, training_data):
    x, y = sliding_windows(training_data, best_seq)
    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = torch.Tensor(np.array(x))
    dataY = torch.Tensor(np.array(y))

    trainX = torch.Tensor(np.array(x[0:train_size]))
    trainY = torch.Tensor(np.array(y[0:train_size]))

    testX = torch.Tensor(np.array(x[train_size:len(x)]))
    testY = torch.Tensor(np.array(y[train_size:len(y)]))

    lstm = LSTM(num_classes, input_size, best_hidden_size, num_layers, best_seq)
    lstm = lstm.to(device)
    early_stop = 50
    least_loss = np.inf
    criterion = torch.nn.MSELoss()    
    optimizer = torch.optim.Adam(lstm.parameters(), lr=best_lr)

    # Train the model
    for epoch in range(num_epochs):
        trainX = trainX.to(device)
        trainY = trainY.to(device)
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        
        if least_loss > loss.item():
            least_loss = loss.item()
            early_stop = 50
        else:
            early_stop -= 1

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        
        if early_stop <= 0:
            print("Epoch: %d, loss: %1.5f, best_loss: %1.5f" % (epoch, loss.item(), least_loss))
            break

    lstm.eval()
    dataX = dataX.to(device)
    dataY = dataY.to(device)
    train_predict = lstm(dataX)

    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.cpu().data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)


    lstm.eval()
    testX = testX.to(device)
    testY = testY.to(device)
    test_predictions = lstm(testX)
    test_loss = criterion(test_predictions, testY)
    print(f"Test Loss for model trained {epoch} epochs: {test_loss.item()}")


    lstm.eval()
    dataX = dataX.to(device)
    train_predict = lstm(dataX)

    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.cpu().data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)

    #Plot the Predictions
    plt.axvline(x=train_size, c='r', linestyle='--')

    plt.plot(dataY_plot, label="True")
    plt.plot(data_predict, label="Predicted")
    plt.legend()
    plt.suptitle('Time-Series Prediction')
    plt.title(f"Test loss: {test_loss.item()}")
    plt.show()


    return lstm


def test_model(lstm, dataY):
    response = requests.get(f"http://144.39.204.242:11236/evr/leviton/evr?limit={best_seq * 85 * 2}")
    usage = response.json()
    ten_minute_averages = {"timestamp": [], "power": [], "isDay": [], "hour": []}
    time_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    for i in range(0, len(usage['data']), 85):
        values = usage['data'][i:i+120]
        power = [value['power'] for value in values]
        ten_minute_averages["timestamp"].append(values[0]['timestamp'])
        ten_minute_averages["power"].append(np.average(power) * 1000)
        test_date = datetime.datetime.strptime(values[0]['timestamp'], time_format)
        test_date = test_date - datetime.timedelta(hours=6)
        isDay =  1 if test_date.hour >= 7 and test_date.hour <= 19 else 0 # From 7:00 AM to 7 PM is day
        hour = test_date.hour
        ten_minute_averages["isDay"].append(isDay)
        ten_minute_averages["hour"].append(hour)
    df = pd.DataFrame(columns=["timestamp", "power", "isDay", "hour"])    
    temp_df = pd.DataFrame(ten_minute_averages)
    df = pd.concat([df, temp_df], ignore_index=True)
    test_training_set = df.iloc[:, 1:5].values


    test_training_set = sc.fit_transform(test_training_set)
    testNewX, testNewY = sliding_windows(test_training_set, best_seq)
    testDataX = Variable(torch.Tensor(np.array(testNewX)))
    testDataY = Variable(torch.Tensor(np.array(testNewY)))

    print(testDataX.shape)

    lstm.eval()
    test_predict = lstm(testDataX.to(device))

    data_predict = test_predict.cpu().data.numpy()
    dataY_test_plot = dataY.cpu().data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(testDataY)

    #Plot the Predictions
    plt.axvline(x=0, c='r', linestyle='--')

    power = [i[0] for i in dataY_plot]
    power_predict = [i[0] for i in data_predict]

    plt.plot(power, label="True")
    plt.plot(power_predict, label="Predicted")
    plt.legend()
    plt.suptitle('Time-Series Prediction')
    plt.show()



if __name__ == "__main__":
    

# =============== HYPERPARAMETERS ===============
    
    num_epochs = 2000
    learning_rates = [0.01, 0.001]
    seq_lengths = [52, 58, 64, 70, 76, 82]
    hidden_sizes = [8, 16, 32]
    device = "cuda"
    best_seq = 0
    best_hidden_size = 0
    best_lr = 0
    lowest_loss = np.inf
    num_classes = 3
    input_size = 3
    num_layers = 1

# ===============================================

    training_set = get_csv()

    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)
    lowest_loss, best_seq, best_hidden_size, best_lr = train_best_model(learning_rates, seq_lengths, hidden_sizes, num_classes, input_size, num_layers, sc, training_data)


    model = train_single_best(lowest_loss, best_seq, best_hidden_size, best_lr, training_set)
