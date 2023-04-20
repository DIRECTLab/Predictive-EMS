import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import random
from tqdm import tqdm

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
    


def generate_random_data(n):
    SOS_token = np.array([2])
    EOS_token = np.array([3])
    length = 8

    data = []

    # 1,1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)

        X[start::2] = 1

        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        X = np.concatenate((SOS_token, X, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))

        data.append([X, y])

    np.random.shuffle(data)

    return data


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

    model.train()
    total_loss = 0

    for batch in dataloader:
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


def validation_loop(model, criterion, dataloader):
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

    return total_loss / len(dataloader)

def fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs):

    train_loss_list, validation_loss_list = [], []

    print("Training and Validating the Model")

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(model, optimizer, criterion, train_dataloader)
        train_loss_list.append(train_loss)

        validation_loss = validation_loop(model, criterion, val_dataloader)
        validation_loss_list.append(validation_loss)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}\n")

    return train_loss_list, validation_loss_list                   
        

def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
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

    train_data = generate_random_data(9000)
    val_data = generate_random_data(3000)

    train_dataloader = batchify_data(train_data)
    val_dataloader = batchify_data(val_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(num_tokens=4, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()


    epochs = 1
    fit(model, optimizer, criterion, train_dataloader, val_dataloader, epochs)


    examples = [
        # torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
        torch.tensor([[2, 0.1, 0.2, 0.3, 0.4, 0.5, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
        # torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
    ]

    for idx, example in enumerate(examples):
        result = predict(model, example)
        print(f"Example {idx}")
        print(f"Input: {example.view(-1).tolist()[1:-1]}")
        print(f"Continuation: {result[1:-1]}")
        print()


