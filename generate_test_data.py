from cva import *
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import qmc

from config import RANDOM_SEED, CVA_TEST_DATASET

def make_lhs_parameter_sample(n_samples, seed=RANDOM_SEED):

    lower_bound = np.array([0.4, 0.75, 0.15, 0.10, 0, -0.75])
    upper_bound = np.array([0.9, 1.4, 0.8, 0.50, 0.1, 0.75])

    sampler = qmc.LatinHypercube(d=6, seed=seed)
    unit_sample = sampler.random(n=n_samples)
    scaled_sample = qmc.scale(unit_sample, lower_bound, upper_bound)

    return pd.DataFrame(scaled_sample, columns=["L_over_V0", "S0_over_K", "sigma_s", "sigma_v", "mu_v", "rho"])


parameter_df = make_lhs_parameter_sample(n_samples=10000, seed=RANDOM_SEED)
print("Number of parameter sets:", len(parameter_df))


if os.path.exists(CVA_TEST_DATASET):
    print(f"Timing dataset already exists at {CVA_TEST_DATASET}. Loading...")
    control_timing_df = pd.read_csv(CVA_TEST_DATASET)

else:
    results = []
    for n_paths in [100000]:
        for n_steps in [52]:
            abs_se = []
            rel_se = []
            elapsed = []
            rows = []

            for i, row in tqdm(parameter_df.iterrows(), total=len(parameter_df)):
                L_over_V0 = row["L_over_V0"]
                S0_over_K = row["S0_over_K"]
                sigma_s = row["sigma_s"]
                sigma_v = row["sigma_v"]
                mu_v = row["mu_v"]
                rho = row["rho"]

                start = time.perf_counter()

                cva, std_error = black_cox_mc_control(
                    L_over_V0=L_over_V0,
                    S0_over_K=S0_over_K,
                    sigma_s=sigma_s,
                    sigma_v=sigma_v,
                    mu_v=mu_v,
                    rho=rho,
                    V0=100,
                    S0=100,
                    T=1,
                    r_s=0.03,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    recovery_rate=0.4,
                    seed=RANDOM_SEED + i,
                )

                end = time.perf_counter()
                elapsed_time = end - start

                if np.isfinite(std_error):
                    abs_se.append(std_error)
                if np.isfinite(std_error) and np.isfinite(cva) and abs(cva) > 1e-14:
                    rel_se.append(std_error / cva)
                elapsed.append(elapsed_time)

                if n_steps == 52 and n_paths == 100000:
                    rows.append({
                        "L_over_V0": L_over_V0,
                        "S0_over_K": S0_over_K,
                        "sigma_s": sigma_s,
                        "sigma_v": sigma_v,
                        "mu_v": mu_v,
                        "rho": rho,
                        "CVA": cva,
                        "standard_error": std_error,
                        "relative_standard_error": std_error / cva if cva != 0 else None,
                        "elapsed_time_seconds": elapsed_time,
                        "n_paths": n_paths,
                        "n_steps": n_steps,
                        "seed": RANDOM_SEED + i,
                    })
            '''results.append({
            "n_paths": n_paths,
            "n_steps": n_steps,
            "mean_abs_se": np.mean(abs_se),
            "median_abs_se": np.median(abs_se),
            "mean_rel_se": np.mean(rel_se),
            "median_rel_se": np.median(rel_se),
            "mean_time": np.mean(elapsed),
        })'''

    control_timing_df = pd.DataFrame(rows)
    control_timing_df.to_csv(CVA_TEST_DATASET, index=False)
    print(f"Saved to {CVA_TEST_DATASET}")