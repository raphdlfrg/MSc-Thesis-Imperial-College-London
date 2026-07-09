from pathlib import Path
import os

if os.path.exists("/content"):
    # Google Colab
    ROOT = Path("/content/drive/MyDrive/Datasets")
else:
    # Local machine
    ROOT = Path("data")

CVA_DATASET = ROOT / "black_cox_cva_control_timing_results.csv"
NN_TRAINING_PATH = ROOT / "nn_training.csv"
NN_PREDICTIONS_PATH = ROOT / "nn_predictions.csv"


# ---------------------------------------
# Synthetic Dataset Generation Parameters 
# ---------------------------------------

N_LHS_SAMPLES = 200000
RANDOM_SEED = 42

# ---------------------------------------
# Training neural network parameters
# ---------------------------------------

VALIDATION_SIZE = 0.10
