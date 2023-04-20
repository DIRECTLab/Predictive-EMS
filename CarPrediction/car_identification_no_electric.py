import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision import transforms
from random import randint
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
import numpy as np
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision


class CarDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, train=False):
        self.cars = pd.read_csv(csv_file)

        self.image_dir = image_dir
        self.transform = transform
        self.label_encoder = preprocessing.LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.cars['label'])

        np.save("classes.npy", self.label_encoder.classes_)
        self.is_train = train
        self.other_transforms = transforms.Compose([transforms.Resize([128, 128], antialias=True)])
    
    def __len__(self):
        return len(self.cars)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        if ".png" in self.cars.iloc[index, 1] or ".jpg" in self.cars.iloc[index, 1] or ".jpeg" in self.cars.iloc[index, 1]:
            image_name = self.cars.iloc[index, 1]
        else:
            image_name = os.path.join(self.image_dir, self.cars.iloc[index, 1]) + '.jpg'
        
        if not os.path.isfile(image_name):
            random_file = randint(1, 16)
            if random_file < 10:
                random_file = f"0{random_file}"
            
            image_name = os.path.join(self.image_dir, self.cars.iloc[index, 1] + f"_{random_file}" + ".jpg")

            if not os.path.isfile(image_name):
                raise Exception(f"{image_name} does not exist (at index {index})")
        
        image = read_image(image_name, ImageReadMode.RGB)

        image /= 255.0
        label = self.encoded_labels[index]

        if self.transform and self.is_train:
            image = self.transform(image)
        else:
            image = self.other_transforms(image)
        
        return (image, label)
            

def get_data(train_csv, valid_csv, test_csv, image_dir):
    composed = transforms.Compose([
        transforms.Resize([128, 128], antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60)
    ])


    train_dataset = CarDataset(csv_file=train_csv, image_dir=image_dir, transform=composed, train=True)
    valid_dataset = CarDataset(csv_file=valid_csv, image_dir=image_dir, transform=composed)
    test_dataset = CarDataset(csv_file=test_csv, image_dir=image_dir, transform=composed)

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = get_data("no_electric_train.csv", "no_electric_valid.csv", "no_electric_test.csv", "data/all-data")

    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, 16)
    test_loader = DataLoader(test_dataset, 16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.RegNet_X_400MF_Weights().to(device) 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    epochs = 10000
    max_early_stop = 20
    early_stop = 20
    train_epoch_losses = []
    valid_epoch_losses = []
    valid_epoch_accuracy = []
    valid_epoch_f1 = []
    last_accuracy = 0

    for epoch in range(epochs):
        loop = tqdm(train_loader)

        model.train()
        epoch_loss = 0

        for i, (x, y) in enumerate(loop):
            y = y.type(torch.LongTensor)
            y = y.to(device)
            x = x.type(torch.FloatTensor)        
            x = x.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
        
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=(epoch_loss / (i + 1)))
        
        train_epoch_losses.append(epoch_loss / len(train_loader))

        # ========================================= Validation =========================================
        loop_validation = tqdm(valid_loader)
        model.eval()
        temp_loss = 0
        pred = []
        labels = []
        
        for i, (x, y) in enumerate(loop_validation):
            y = y.type(torch.LongTensor)
            y = y.to(device)
            x = x.type(torch.FloatTensor)        
            x = x.to(device)

            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, y)
            temp_loss += loss.item()

            pred.extend(predictions.view(-1).cpu().detach().numpy())
            labels.extend(y.view(-1).cpu().detach().numpy())

            accuracy = accuracy_score(labels, pred)
            f1 = f1_score(labels, pred, average="macro")

            loop_validation.set_description(f"Validation Epoch [{epoch + 1}/{epochs}]")
            loop_validation.set_postfix_str(f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(accuracy, 3)} and F1 score: {round(f1, 3)}")
        

        valid_accuracy = accuracy_score(labels, pred)
        valid_f1 = f1_score(labels, pred, average='macro')
        valid_epoch_f1.append(valid_f1)
        valid_epoch_losses.append(temp_loss / len(valid_loader))
        valid_epoch_accuracy.append(valid_accuracy)
        scheduler.step(temp_loss / len(valid_loader))
        
        # Early Stopping Criteria
        if (last_accuracy < valid_accuracy):
            print(f"Accuracy improved from {last_accuracy} -> {valid_accuracy}. Saving Model")
            last_accuracy = valid_accuracy
            torch.save(model.state_dict(), f'./no_electric_model.bin')
            early_stop = max_early_stop
        else:
            early_stop -= 1


        if early_stop <= 0:
            print(f"Early stopping because accuracy hasn't improved for the last {max_early_stop} epochs")
            break

    model.eval()
    loop_test = tqdm(test_loader)
    temp_loss = 0
    pred = []
    labels = []

    for i, (x, y) in enumerate(loop_test):
        y = y.type(torch.LongTensor)
        y = y.to(device)
        x = x.type(torch.FloatTensor)        
        x = x.to(device)

        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        temp_loss += loss.item()
        pred.extend(predictions.view(-1).cpu().detach().numpy())
        labels.extend(y.view(-1).cpu().detach().numpy())

        accuracy = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred, average='macro')

        loop_test.set_description(f"Test Epoch")
        loop_test.set_postfix_str(f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(accuracy, 3)}, F1: {round(f1, 3)}")

    test_loss = temp_loss / len(test_loader)
    test_accuracy = accuracy_score(labels, pred)
    test_f1 = f1_score(labels, pred, average='macro')
        
    print(f"Test Loss: {test_loss} with an accuracy of {test_accuracy}")
    fig, ax = plt.subplots(1, 3, figsize=(10, 7))
    plt.suptitle(f"RegNet")
    ax[0].plot(train_epoch_losses, label="Train Loss")
    ax[0].plot(valid_epoch_losses, label="Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].legend()

    ax[1].plot(valid_epoch_accuracy, label="Validation Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy over Epochs")
    ax[1].legend()

    ax[2].plot(valid_epoch_f1, label="Validation F1")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("F1 Score")
    ax[2].set_title("F1 Score over Epochs")
    ax[2].legend()

    plt.savefig(f'./no_electric_car_identification.jpg')

    train_epoch_losses = np.array(train_epoch_losses)
    valid_epoch_losses = np.array(valid_epoch_losses)
    valid_epoch_accuracy = np.array(valid_epoch_accuracy)
    valid_epoch_f1 = np.array(valid_epoch_f1)

    np.savetxt('train_epoch_losses_regnet.txt', train_epoch_losses, delimiter=',')
    np.savetxt('valid_epoch_losses_regnet.txt', valid_epoch_losses, delimiter=',')
    np.savetxt('valid_epoch_accuracy_regnet.txt', valid_epoch_accuracy, delimiter=',')
    np.savetxt('valid_epoch_f1_regnet.txt', valid_epoch_f1, delimiter=',')
    np.savetxt('test_loss_accuracy_f1_regnet.txt', np.array([test_loss, test_accuracy, test_f1]))