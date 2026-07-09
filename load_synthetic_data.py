from cva import *

import sys
print(sys.executable)

import numpy as np
import os 
import pandas as pd
from itertools import product
from tqdm import tqdm
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from scipy.stats import qmc

#
def make_lhs_samples(n_samples, seed=42):
    LHS_sampler = qmc.LatinHypercube(d=6, seed=seed)  # 6 parameters: L_over_V0, S0, sigma_v, sigma_s, rho
    LHS_sample = LHS_sampler.random(n=n_samples, workers=-1)
    lower_bound = np.array([0.4, 0.7, 0.1, 0.1, -0.02, -0.75])
    upper_bound = np.array([0.8, 1.3, 0.6, 0.6, 0.08, 0.75])
    scaled_sample = qmc.scale(LHS_sample, lower_bound, upper_bound)
    df_params = pd.DataFrame(scaled_sample, columns=["L_over_V0", "S0_over_K", "sigma_v", "sigma_s", "mu_v", "rho"])
    return df_params




# training grid
'''
L_over_V0_values = [0.05, 0.20, 0.5]

S0_values = [80, 100, 120]

sigma_v_values = [0.1, 0.4, 0.9]

sigma_s_values = [0.1, 0.35, 0.8]

rho_values = [-0.95, -0.5, 0.0, 0.5]

training_grid = list(product(
    L_over_V0_values,
    S0_values,
    sigma_v_values,
    sigma_s_values,
    rho_values
))

print(len(training_grid))'''


TRAIN_PATH = "data/cva_training_dataset.csv"

if os.path.exists(TRAIN_PATH):
    print(f"Training dataset already exists at {TRAIN_PATH}. Loading...")
    
else:
    df_params = make_lhs_samples(n_samples=200, seed=42)

    rows = []

    for _, row in tqdm(df_params.iterrows(), total=len(df_params)):
        L_over_V0 = row["L_over_V0"]
        S0_over_K = row["S0_over_K"]
        sigma_v = row["sigma_v"]
        sigma_s = row["sigma_s"]
        mu_v = row["mu_v"]
        rho = row["rho"]
        V0 = 100
        S0 = 100
        K = S0 / S0_over_K
        cva = black_cox_mc_control(
            L_over_V0=L_over_V0,
            S0_over_K=S0_over_K,
            sigma_s=sigma_s,
            sigma_v=sigma_v,
            mu_v=mu_v,
            rho=rho,
            V0=V0,
            S0=S0,
            T=1,
            r_s=0.03,
            n_paths=100000,
            n_steps=500,
            recovery_rate=0.4,
            seed=42
        )
        rows.append({
            "L_over_V0": L_over_V0,
            "S0_over_K": S0_over_K,
            "S0": S0,
            "sigma_v": sigma_v,
            "sigma_s": sigma_s,
            "mu_v": mu_v,
            "rho": rho,
            "CVA": cva
        })

    df_train = pd.DataFrame(rows)
    df_train.to_csv("cva_training_dataset.csv", index=False)

# Validation grid
'''
L_over_V0_val = [0.05, 0.45]

S0_val = [95, 105, 115]

sigma_v_val = [0.15, 0.65, 0.95]

sigma_s_val = [0.15, 0.65, 0.95]

rho_val = [-0.75, -0.25, 0.4, 0.95]

validation_grid = list(product(
    L_over_V0_val,
    S0_val,
    sigma_v_val,
    sigma_s_val,
    rho_val
))

print(len(validation_grid))'''


VALIDATION_PATH = "cva_validation_dataset.csv"
if os.path.exists(VALIDATION_PATH):
    print(f"Validation dataset already exists at {VALIDATION_PATH}. Loading...")
    
else:

    df_params_val = make_lhs_samples(n_samples=100, seed=123)

    rows = []

    for _, row in tqdm(df_params_val.iterrows(), total=len(df_params_val)):
        L_over_V0 = row["L_over_V0"]
        S0_over_K = row["S0_over_K"]
        sigma_v = row["sigma_v"]
        sigma_s = row["sigma_s"]
        mu_v = row["mu_v"]
        rho = row["rho"]
        V0 = 100
        S0 = 100
        K = S0 / S0_over_K
        cva = black_cox_mc_control(
            L_over_V0=L_over_V0,
            S0_over_K=S0_over_K,
            sigma_s=sigma_s,
            sigma_v=sigma_v,
            mu_v=mu_v,
            rho=rho,
            V0=V0,
            S0=S0,
            T=1,
            r_s=0.03,
            n_paths=100000,
            n_steps=500,
            recovery_rate=0.4,
            seed=42
        )

        rows.append({
            "L_over_V0": L_over_V0,
            "S0_over_K": S0_over_K,
            "S0": S0,
            "sigma_v": sigma_v,
            "sigma_s": sigma_s,
            "mu_v": mu_v,
            "rho": rho,
            "CVA": cva
        })

        df_val = pd.DataFrame(rows)
        df_val.to_csv("cva_validation_dataset.csv", index=False)



