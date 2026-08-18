
from config import NN_MODEL_PATH, CVA_DATASET, CVA_TEST_DATASET, SHAP_GLOBAL_IMPORTANCE_PATH, SHAP_GLOBAL_IMPORTANCE_PLOT_PATH, SHAP_WATERFALL_PATH, SHAP_BEESWARM_PATH, VALIDATION_SIZE, RANDOM_SEED, BACKGROUND_SIZE, EXPLANATION_SIZE, NN_SCALERS_PATH, SHAP_DEPENDENCE_PATH
from model import NeuralNetwork 
import torch 
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(NN_MODEL_PATH, weights_only=False, map_location=device)

X_scaler = joblib.load(NN_SCALERS_PATH)["scaler_X"]
Y_scaler = joblib.load(NN_SCALERS_PATH)["scaler_Y"]

model = NeuralNetwork(
    input_size=checkpoint["input_size"],
    hidden_sizes=checkpoint["hidden_sizes"],
    activation_fn=nn.ELU,
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


df = pd.read_csv(CVA_DATASET)

feature_names = df.columns[:6].tolist()

X, Y = df.iloc[:, :6].values, df.iloc[:, 6].values

X_test = pd.read_csv(CVA_TEST_DATASET).iloc[:, :6].values
Y_test = pd.read_csv(CVA_TEST_DATASET).iloc[:, 6].values

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED, shuffle=True)

rng = np.random.default_rng(RANDOM_SEED)

background_indices = rng.choice(len(X_train), size=BACKGROUND_SIZE, replace=False)

explanation_indices = rng.choice(len(X_test), size=EXPLANATION_SIZE, replace=False)

X_background = X_train[background_indices]
X_explanation = X_test[explanation_indices]
Y_explanation = Y_test[explanation_indices]

X_background_tensor = torch.tensor(X_scaler.transform(X_background), dtype=torch.float32, device=device)
X_explanation_tensor = torch.tensor(X_scaler.transform(X_explanation), dtype=torch.float32, device=device)

explainer = shap.DeepExplainer(model, X_background_tensor)

shap_values_scaled = explainer.shap_values(X_explanation_tensor)

print(type(shap_values_scaled))
print(shap_values_scaled.shape)

# A voir, la suite pour tester les shapes c'est pas forcément nécessaire

shap_values_scaled = np.squeeze(shap_values_scaled, axis=-1)


Y_scale = float(Y_scaler.scale_[0])
Y_mean = float(Y_scaler.mean_[0])

shap_values = shap_values_scaled * Y_scale

expected_value_scaled = float(np.asarray(explainer.expected_value).reshape(-1)[0])

expected_value = expected_value_scaled * Y_scale + Y_mean

with torch.no_grad():
    Y_pred_scaled = model(X_explanation_tensor).cpu().numpy().reshape(-1)

predictions_cva = (Y_pred_scaled * Y_scale + Y_mean).reshape(-1)

reconstructed_predictions = (expected_value + shap_values.sum(axis=1))

additivity_error = np.abs(predictions_cva - reconstructed_predictions)

print("Mean additivity error:", additivity_error.mean())
print("Maximum additivity error:", additivity_error.max())

mean_absolute_shap = np.abs(shap_values).mean(axis=0)

global_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_absolute_shap
})

global_importance["Importance_Percent"] = (
    100
    * global_importance["Mean_Absolute_SHAP"]
    / global_importance["Mean_Absolute_SHAP"].sum()
)

global_importance = (global_importance.sort_values("Mean_Absolute_SHAP", ascending=False).reset_index(drop=True))

print("\nGlobal SHAP importance:")
print(global_importance)

global_importance.to_csv(SHAP_GLOBAL_IMPORTANCE_PATH, index=False)

plot_importance = global_importance.sort_values("Mean_Absolute_SHAP", ascending=True)

plt.figure(figsize=(8, 5))

plt.barh(plot_importance["Feature"], plot_importance["Mean_Absolute_SHAP"], color="steelblue")

plt.xlabel("Mean absolute SHAP value (CVA units)")
plt.ylabel("Feature")
plt.title("Global SHAP Feature Importance")
plt.tight_layout()

plt.savefig(SHAP_GLOBAL_IMPORTANCE_PLOT_PATH, dpi=300, bbox_inches="tight")

plt.close()


explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_explanation), expected_value),
    data=X_explanation,
    feature_names=feature_names
)

shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)

plt.xlabel("SHAP value — impact on predicted CVA")
plt.tight_layout()

plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches="tight")

plt.close()


waterfall_index = int(np.argmax(predictions_cva))

print("\nWaterfall observation index:", waterfall_index)

print("Predicted CVA:", predictions_cva[waterfall_index])

print("True CVA:", Y_explanation[waterfall_index])

print("Baseline predicted CVA:", expected_value)

shap.plots.waterfall(explanation[waterfall_index], max_display=len(feature_names), show=False)

plt.tight_layout()

plt.savefig(SHAP_WATERFALL_PATH, dpi=300, bbox_inches="tight")

plt.close()

X_explanation_df = pd.DataFrame(
    X_explanation,
    columns=feature_names
)

shap.dependence_plot(
    ind="L_over_V0",
    shap_values=shap_values,
    features=X_explanation_df,
    interaction_index="sigma_v",
    alpha=0.6,
    dot_size=15
)

plt.title("Dependence of L/V0 contribution on sigma_v")
plt.tight_layout()

plt.savefig(SHAP_DEPENDENCE_PATH, dpi=300, bbox_inches="tight")

plt.close()

print(
    "X scaler mean matches:",
    np.allclose(
        X_train.mean(axis=0),
        X_scaler.mean_,
        rtol=1e-5,
        atol=1e-6
    )
)

print(
    "X scaler scale matches:",
    np.allclose(
        X_train.std(axis=0, ddof=0),
        X_scaler.scale_,
        rtol=1e-5,
        atol=1e-6
    )
)

print(
    "Y scaler mean matches:",
    np.allclose(
        Y_train.mean(),
        Y_scaler.mean_[0],
        rtol=1e-5,
        atol=1e-6
    )
)

print(
    "Y scaler scale matches:",
    np.allclose(
        Y_train.std(ddof=0),
        Y_scaler.scale_[0],
        rtol=1e-5,
        atol=1e-6
    )
)

