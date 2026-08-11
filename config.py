from pathlib import Path
import os

if os.path.exists("/content"):
    # Google Colab
    ROOT = Path("/content/drive/MyDrive/Datasets")
else:
    # Local machine
    ROOT = Path("data")

CVA_DATASET = ROOT / "black_cox_cva_control_timing_results.csv"
CVA_TEST_DATASET = ROOT / "black_cox_cva_control_timing_test_results.csv"

NN_EPOCHS_LOSS_PATH = ROOT / "nn_epochs_loss.csv"
NN_PREDICTIONS_PATH = ROOT / "nn_predictions.csv"

NN_RESULTS_METRICS = ROOT / "nn_results_metrics.csv"
NN_ERROR_BY_CVA = ROOT / "error_by_cva_quantile.csv"
CALIBRATION_BY_DECILE = ROOT / "calibration_by_decile.csv"
VALIDATION_PLOTS_PATH = ROOT / "validation_plots.png"

NN_ARCHITECTURE_RESULTS_PATH = ROOT / "nn_architecture_results.csv"
NN_MODEL_PATH = ROOT / "nn_model.pth"
NN_SCALERS_PATH = ROOT / "nn_scalers.pkl"
# ---------------------------------------
# Synthetic Dataset Generation Parameters 
# ---------------------------------------

N_LHS_SAMPLES = 200000
RANDOM_SEED = 42

# ---------------------------------------
# Training neural network parameters
# ---------------------------------------

VALIDATION_SIZE = 0.10
PATIENCE_EARLY_STOPPING = 10

# ---------------------------------------
# SHAP Parameters
# ---------------------------------------

BACKGROUND_SIZE = 500
EXPLANATION_SIZE = 5000