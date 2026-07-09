import os
import torch
from torch import nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv("data/black_cox_cva_control_timing_results.csv")

X, Y = df.iloc[:, :6].values, df.iloc[:, 6].values


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1, random_state=SEED, shuffle=True)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_validation = scaler_X.transform(X_validation)

scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
Y_validation = scaler_Y.transform(Y_validation.reshape(-1, 1))

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_validation = torch.tensor(X_validation, dtype=torch.float32)
Y_validation = torch.tensor(Y_validation, dtype=torch.float32)

training_dataset = TensorDataset(X_train, Y_train)
validation_dataset = TensorDataset(X_validation, Y_validation)

print(X_train.shape, Y_train.shape)
print(X_validation.shape, Y_validation.shape)

train_loader = DataLoader(
    training_dataset,
    batch_size=1024,
    shuffle=True,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=4096,
    shuffle=False,
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


epochs = 1000

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for i, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(train_loader) 

def validate(model, validation_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in validation_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            running_loss += loss.item()
    return running_loss / len(validation_loader)

training_losses = []
validation_losses = []

for epoch in tqdm(range(epochs), desc="Training"):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    validation_loss = validate(model, validation_loader, loss_fn)
    training_losses.append(train_loss)
    validation_losses.append(validation_loss)

    if epoch % 512 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}")


loss_df = pd.DataFrame({
    "epoch": range(1, epochs + 1),
    "train_loss": training_losses,
    "validation_loss": validation_losses
})

loss_df.to_csv("data/nn_training.csv", index=False)

prediction_df = pd.DataFrame({
    "Predicted_CVA": model(X_validation.to(device)).detach().cpu().numpy() * scaler_Y.scale_ + scaler_Y.mean_,
    "True_CVA": Y_validation.detach().cpu().numpy() * scaler_Y.scale_ + scaler_Y.mean_
})

prediction_df.to_csv("data/nn_predictions.csv", index=False)