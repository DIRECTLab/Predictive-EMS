import torch
import torch.nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io, transform
from torchvision import transforms, utils
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
    def __init__(self, csv_file, image_dir, transform=None):
        self.cars = pd.read_csv(csv_file)
        print(len(self.cars))
        self.image_dir = image_dir
        self.transform = transform
        self.labelEncoder = preprocessing.LabelEncoder()
        self.encoded_labels = self.labelEncoder.fit_transform(self.cars['label'])

        np.save("classes.npy", self.labelEncoder.classes_)
        self.is_train = False
        self.other_transforms = transforms.Compose([transforms.Resize([256, 256]),])
    
    def __len__(self):
        return len(self.cars)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        print(self.cars.iloc[index, 1])

        #TODO: Test this. Might work like this. 
        if ".png" in self.cars.iloc[index, 1] or ".jpg" in self.cars.iloc[index, 1]:
            image_name = self.cars.iloc[index, 1]
        else:
            image_name = os.path.join(self.image_dir, self.cars.iloc[index, 1]) + '.jpg'
        # print(image_name)
        # print(self.cars.iloc[index, 1])
        
        if not os.path.isfile(image_name):
            random_file = randint(1, 16)
            if random_file < 10:
                random_file = f"0{random_file}"
            image_name = os.path.join(self.image_dir, self.cars.iloc[index, 1] + f"_{random_file}" + '.jpg')
            if not os.path.isfile(image_name):
                print(index)
                print(self.cars[index])
                raise Exception(f"{image_name} does not exist")

        image = read_image(image_name, ImageReadMode.RGB)
        # label = self.cars.iloc[index, 2]

        image = image / 255.0
        label = self.encoded_labels[index]
            
        if self.transform and self.is_train:
            image = self.transform(image)
        else:
            image = self.other_transforms(image)

        

        return (image, label)