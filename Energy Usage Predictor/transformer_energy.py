import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt

# Transformer adapted from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout):
        super().__init__()


        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout=dropout, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)

        self.out = nn.Linear(dim_model, num_tokens)

    def get_target_mask(self, size):
        # Generates a squeare matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Converts zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Converts all ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix, pad_token):
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


    def forward(self, src, target, target_mask=None, src_pad_mask=None, target_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        target = self.embedding(target) * math.sqrt(self.dim_model)

        src = self.positional_encoder(src)
        target = self.positional_encoder(target)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        tranformer_out = self.transformer(src, target, tgt_mask=target_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=target_pad_mask)
        out = self.out(tranformer_out)

        return out




class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
    
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
# This will be the thing that generates the "true" energy usage. Will eventually be based on historical data instead. (Get the past 4 days at the leviton)
def generate_energy_usage():
    today = datetime.datetime.now()

    date_start = datetime.date.fromtimestamp(1678431600)
    
    delta_end = datetime.timedelta(days=2)
    date_end = today - delta_end
    # date_start = today - delta_start

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

        ten_minute_averages.append(math.ceil((np.average(power) * 1000) / 1000) + 100) # Shift it so there are no negative values


    return ten_minute_averages


def generate_val_data(seq_length):
    SOS_token = np.array([301])
    EOS_token = np.array([302])

    data = []

    ten_minute_averages = generate_energy_usage()

    for i in range(len(ten_minute_averages) - seq_length):
        X = np.concatenate((SOS_token, ten_minute_averages[i:i+seq_length], EOS_token))
        y = np.concatenate((SOS_token, ten_minute_averages[i:i+seq_length], EOS_token))
        data.append([X, y])
    
    
    return data


def generate_data(seq_length):
    SOS_token = np.array([301])
    EOS_token = np.array([302])

    data = []

    csv_data = pd.read_csv("power_usage.csv")
    csv_data = csv_data.iloc[:, 2].values
    csv_data = [math.ceil(i / 1000) + 100 for i in csv_data if float(i) > -100_000 and float(i) < 200_000] # Most likely errors in device if reading < -100,000. Then shift it up by 100 to have only positive values
    for i in range(len(csv_data) - seq_length):
        X = np.concatenate((SOS_token, csv_data[i:i+seq_length], EOS_token))
        y = np.concatenate((SOS_token, csv_data[i:i+seq_length], EOS_token))
        data.append([X, y])

    np.random.shuffle(data)

    return data


def get_test_data(seq_length):
    response = requests.get(f"http://144.39.204.242:11236/evr/leviton/evr?limit={2 * seq_length * 100}")
    usage = response.json()
    ten_minute_averages = []
    SOS_token = np.array([301])
    EOS_token = np.array([302])
    test_data = []
    true_data = []

    # Chunk it into 10 minute increments. Change this to 120 in a week or so
    for i in range(0, len(usage['data']), 85):
        values = usage['data'][i:i+120]
        power = [value['power'] for value in values]

        ten_minute_averages.append(math.ceil((np.average(power) * 1000) / 1000) + 100) # Shift it so there are no negative values

    for i in range(len(ten_minute_averages) - 2*seq_length):
        X = np.concatenate((SOS_token, ten_minute_averages[i:i+seq_length], EOS_token))
        # y = np.concatenate((SOS_token, ten_minute_averages[i:i+seq_length], EOS_token))
        true_data = ten_minute_averages[i+seq_length:i+seq_length*2]
        test_data.append([X])
        # break # This will only give a single example case. Could be what we care about
    
    for i in range(len(true_data)):
        true_data[i] -= 100
    return test_data, true_data


def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


def train_loop(model, optimizer, criterion, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device ="cpu"


    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, leave=False, colour="red"):
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the target by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        target_mask = model.get_target_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and target_mask
        pred = model(X, y_input, target_mask=target_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = criterion(pred, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
    
    return total_loss / len(dataloader)


def validation_loop(model, criterion, dataloader, scheduler):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            target_mask = model.get_target_mask(sequence_length).to(device)

            # Standard validation but with mask
            pred = model(X, y_input, target_mask)

            # Permute the predictions to have batch size first
            pred = pred.permute(1, 2, 0)
            loss = criterion(pred, y_expected)

            total_loss += loss.detach().item()

    scheduler.step(total_loss / len(dataloader))

    return total_loss / len(dataloader)

def fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs, scheduler):

    train_loss_list, validation_loss_list = [], []

    best_loss_so_far = np.inf
    early_stop = 20
    max_early_stop = 20
    print("Training and Validating the Model")

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(model, optimizer, criterion, train_dataloader)
        train_loss_list.append(train_loss)

        validation_loss = validation_loop(model, criterion, val_dataloader, scheduler)
        validation_loss_list.append(validation_loss)

        if validation_loss < best_loss_so_far:
            best_loss_so_far = validation_loss
            early_stop = max_early_stop
        else:
            early_stop -= 1

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}\n")

        if early_stop <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    return train_loss_list, validation_loss_list                   
        

def predict(model, input_sequence, max_length=65, SOS_token=301, EOS_token=302):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_target_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()



if __name__ == "__main__":

    # TODO: Grid serach over transformer parameters
    # seq_lengths = [30, 40, 60, 80, 144]
    # lrs = [0.01, 0.001, 0.0001]
    # num_encoder_layers = [8, 10]
    # num_decoder_layers = [8, 10]
    # num_heads = [8, 16, 32]
    seq_length = 40
    lr = 0.001
    num_head = 16
    num_encoder_layer = 8
    num_decoder_layer = 8
    length_of_prediction = 18
    epochs = 500

    # seq_lengths = [40, 60, 80, 144]
    # lrs = [0.01, 0.001, 0.0001]
    # num_encoder_layers = [8, 16]
    # num_decoder_layers = [8, 16]
    # num_heads = [8, 16]

   
    # num_tokens is each step of 1000 between -100,000 and 200,000. This equates to 80 + 2 tokens (for EOS and SOS)
    num_tokens = int(300_000 / 1000 + 3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device ="cpu"
    best_loss = np.inf

    train_data = generate_data(seq_length)
    val_data = generate_val_data(seq_length)

    train_dataloader = batchify_data(train_data)
    val_dataloader = batchify_data(val_data)


       
    print(f" ===================== Starting LR: {lr}, Num Heads: {num_head}, Num Encoder Layer: {num_encoder_layer}, Num Decoder Layer: {num_decoder_layer}, Seq Length: {seq_length} =====================")
    model = Transformer(num_tokens=num_tokens, dim_model=512, num_heads=num_head, num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)

    train_loss, validation_loss = fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs, scheduler)
    # training_losses.append(train_loss)
    # validation_losses.append(validation_loss)

    if validation_loss[-1] < best_loss:
        best_loss = validation_loss[-1]

    results = []
    examples, true_data = get_test_data(seq_length)
    examples = examples[0]


    for idx, example in enumerate(examples):
        # Predict 6 hours
        result = predict(model, torch.tensor(np.array([example]), dtype=torch.long, device=device), max_length=length_of_prediction)
        print(f"Example {idx}")
        print(f"Input: {example[1:-1]}")
        print(f"Continuation: {result[1:-1]}")

        for i in range(1, len(result)):
            result[i] -= 100
        results.append(result[1:-1])

        plt.plot(result[1:-1])
        plt.plot(true_data[:length_of_prediction])
        plt.xlabel("10 minute durations")
        plt.ylabel("kW usage")
        plt.title(f"Starting LR: {lr}, Num Heads: {num_head}, Num Encoder Layer: {num_encoder_layer}, Num Decoder Layer: {num_decoder_layer}, Seq Length: {seq_length}")
        plt.savefig(f"energy_LR_{lr}_Num Heads_{num_head}_Num Encoder Layer_{num_encoder_layer}_Num Decoder Layer_{num_decoder_layer}_Seq Length_{seq_length}.jpg")
        plt.close()
                            
    #print(f"The Best Network:\n \tLR: {best_lr}\n\tNum Heads: {best_num_head}\n\tNum Encoder Layers: {best_num_encoder_layer}\n\tNum Decoder Layers: {best_num_decoder_layer}, Sequence Length: {best_seq_length} ")

    
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].plot(results[0], label="Predictions")
    ax[0].plot(true_data[:length_of_prediction], label="True Data")
    ax[0].legend()
    ax[0].set_xlabel("10 minute durations")
    ax[0].set_ylabel("kW usage")
    ax[0].set_title("Energy Predictions vs. Data for next 3 hours")

    ax[1].plot(train_loss, label="Train Loss")
    ax[1].plot(validation_loss, label="Validation Loss")
    ax[1].legend()
    
    plt.savefig("energy_predictions.jpg")        
    torch.save(model.state_dict(), "transformer_energy_predictor.pth")


    with open("training_losses.txt", "w") as f:
        for loss in train_loss:
            f.write("" + str(loss) + "\n")

    with open("validation_losses.txt", "w") as f:
        for loss in validation_loss:
            f.write("" + str(loss) + "\n")

