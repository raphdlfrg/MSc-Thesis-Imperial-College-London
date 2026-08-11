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
from itertools import product
import joblib

from config import CVA_DATASET, NN_PREDICTIONS_PATH, NN_EPOCHS_LOSS_PATH, PATIENCE_EARLY_STOPPING, VALIDATION_SIZE, RANDOM_SEED, NN_ARCHITECTURE_RESULTS_PATH, NN_MODEL_PATH, NN_SCALERS_PATH

from model import NeuralNetwork


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

def train_model(model, train_loader, validation_loader, loss_fn=nn.MSELoss(), lr=1e-3, delta=1e-6, max_epochs=200):
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
        if validation_loss < best_validation_loss - delta:
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
        "best_validation_rmse": np.sqrt(best_validation_loss),
        "best_epoch": int(np.argmin(validation_losses)) + 1,
        "epochs_trained": len(training_losses)
    }


name = "3_layers_128_64_32"

learning_rates = [3e-4, 1e-3, 5e-3, 1e-2]
batch_sizes = [128, 256, 512]

hidden_sizes = [128, 64, 32]
seed = RANDOM_SEED


results = []
trained_models = {}
training_histories = {}

for lr, batch_size in product(learning_rates, batch_sizes):
    print(f"Training model with architecture: {hidden_sizes}, learning rate: {lr}, batch size: {batch_size}")

    config_id = f"hidden_sizes_{hidden_sizes}_lr_{lr}_batch_size_{batch_size}"

    #We reset the random seeds for reproducibility (so each architecture starts from the same initial weights)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


    train_loader = DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4096,
        shuffle=False,
    )


    model = NeuralNetwork(input_size=6, hidden_sizes=hidden_sizes).to(device)

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        lr=lr
    )

    number_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    results.append({
        "config_id": config_id,
        "name": name,
        "hidden_sizes": str(hidden_sizes),
        "activation": "elu",
        "learning_rate": lr,
        "batch_size": batch_size,
        "parameters": number_parameters,
        "best_epoch": training_result["best_epoch"],
        "epochs_trained": training_result["epochs_trained"],
        "best_validation_loss":
            training_result["best_validation_loss"],
        "best_validation_rmse":
            training_result["best_validation_rmse"],
        "number_parameters": number_parameters
    })

    trained_models[config_id] = training_result["model"]
    training_histories[config_id] = training_result["history"]

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="best_validation_rmse").reset_index(drop=True)

results_df.to_csv(NN_ARCHITECTURE_RESULTS_PATH, index=False)

best_config = results_df.loc[0, "config_id"]
best_model = trained_models[best_config]

checkpoint = {
    "model_state_dict": best_model.state_dict(),
    "input_size": 6,
    "hidden_sizes": hidden_sizes,
    "activation": "elu",
    "learning_rate": results_df.loc[0, "learning_rate"],
    "batch_size": results_df.loc[0, "batch_size"],
}

torch.save(checkpoint, NN_MODEL_PATH)

joblib.dump({ "scaler_X": scaler_X, "scaler_Y": scaler_Y }, NN_SCALERS_PATH)


print(f"Best combination: {best_config}")


best_model_training_losses = training_histories[best_config]["training_losses"]
best_model_validation_losses = training_histories[best_config]["validation_losses"]


loss_df = pd.DataFrame({
    "epoch": range(1, len(best_model_training_losses) + 1),
    "train_loss": best_model_training_losses,
    "validation_loss": best_model_validation_losses
})

loss_df.to_csv(NN_EPOCHS_LOSS_PATH, index=False)

best_model.eval()

with torch.no_grad():
    predicted_scaled = best_model(X_validation.to(device)).cpu().numpy()

predicted_cva = (predicted_scaled * scaler_Y.scale_ + scaler_Y.mean_).ravel()

true_cva = (Y_validation.cpu().numpy() * scaler_Y.scale_ + scaler_Y.mean_).ravel()

prediction_df = pd.DataFrame({
    "Predicted_CVA": predicted_cva,
    "True_CVA": true_cva,
    "Residuals": predicted_cva - true_cva,
    "Absolute_errors": np.abs(predicted_cva - true_cva),
    "Squared_errors": (predicted_cva - true_cva) ** 2
})

prediction_df.to_csv(NN_PREDICTIONS_PATH, index=False)
