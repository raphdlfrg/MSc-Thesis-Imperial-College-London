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
import copy

from config import CVA_DATASET, NN_PREDICTIONS_PATH, NN_TRAINING_PATH, PATIENCE_EARLY_STOPPING, VALIDATION_SIZE, RANDOM_SEED

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

df = pd.read_csv(CVA_DATASET)

X, Y = df.iloc[:, :6].values, df.iloc[:, 6].values


X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED, shuffle=True)

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
    batch_size=512,
    shuffle=True,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=4096,
    shuffle=False,
)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=(64, 64), activation_fn=nn.ELU):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        predictions = self.network(x)
        return predictions

    

model = NeuralNetwork().to(device)
print(model)




epochs = 200

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

def train_model(model, train_loader, validation_loader, loss_fn=nn.MSELoss(), lr=1e-3, max_epochs=200):
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    loss_fn = loss_fn

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(max_epochs), desc="Training"):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        validation_loss = validate(model, validation_loader, loss_fn)
        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE_EARLY_STOPPING:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

        if epoch % 20 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}")

    
    model.load_state_dict(best_model_state)

    return {
        "model": model,
        "history": {"training_losses": training_losses, "validation_losses": validation_losses},
        "best_validation_loss": best_validation_loss,
        "best_model_rmse": np.sqrt(best_validation_loss),
        "best_epoch": int(np.argmin(validation_losses)) + 1,
        "epochs_trained": len(training_losses)
    }

architectures = [
    {
        "name": "1_layer_64",
        "hidden_sizes": [64],
        "activation": "elu",
        "learning_rate": 1e-3
    },
    {
        "name": "1_layer_128",
        "hidden_sizes": [128],
        "activation": "elu",
        "learning_rate": 1e-3
    },
    {
        "name": "2_layers_64_64",
        "hidden_sizes": [64, 64],
        "activation": "elu",
        "learning_rate": 1e-3
    },
    {
        "name": "2_layers_128_64",
        "hidden_sizes": [128, 64],
        "activation": "elu",
        "learning_rate": 1e-3
    },
    {
        "name": "3_layers_128_64_32",
        "hidden_sizes": [128, 64, 32],
        "activation": "elu",
        "learning_rate": 1e-3
    }
]

results = []
trained_models = {}
training_histories = {}

for config in architectures:
    print(f"Training: {config['name']}")

    #We reset the random seeds for reproducibility (so each architecture starts from the same initial weights)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model = NeuralNetwork(input_size=6, hidden_sizes=config["hidden_sizes"]).to(device)

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        learning_rate=config["learning_rate"],
        max_epochs=200,
        patience=10,
        min_delta=1e-6
    )

    number_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    results.append({
        "name": config["name"],
        "hidden_sizes": str(config["hidden_sizes"]),
        "activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "parameters": number_parameters,
        "best_epoch": training_result["best_epoch"],
        "epochs_trained": training_result["epochs_trained"],
        "best_validation_loss":
            training_result["best_validation_loss"],
        "best_validation_rmse":
            training_result["best_validation_rmse"]
    })

    trained_models[config["name"]] = training_result["model"]
    training_histories[config["name"]] = training_result["history"]

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="best_validation_rmse").reset_index(drop=True)
print(results_df)

best_model_name = results_df.loc[0, "name"]
best_model = trained_models[best_model_name]

print(f"Best architecture: {best_model_name}")


'''
loss_df = pd.DataFrame({
    "epoch": range(1, len(training_losses) + 1),
    "train_loss": training_losses,
    "validation_loss": validation_losses
})

loss_df.to_csv(NN_TRAINING_PATH, index=False)

model.eval()

with torch.no_grad():
    predicted_scaled = model(X_validation.to(device)).cpu().numpy()

predicted_cva = (predicted_scaled * scaler_Y.scale_ + scaler_Y.mean_).ravel()

true_cva = (Y_validation.cpu().numpy() * scaler_Y.scale_ + scaler_Y.mean_).ravel()

prediction_df = pd.DataFrame({
    "Predicted_CVA": predicted_cva,
    "True_CVA": true_cva
})

prediction_df.to_csv(NN_PREDICTIONS_PATH, index=False)
'''