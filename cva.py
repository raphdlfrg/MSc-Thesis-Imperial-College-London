import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import dblquad
import time
from itertools import product
import pandas as pd
from tqdm import tqdm

#-----------------------
# Merton model functions
#-----------------------

def proba_default(V0, r, sigma, T, L):
    d2 = (np.log(L / V0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d2)


def call_option_price(V0, r, sigma, T, L):
    d1 = (np.log(V0 / L) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (V0 * norm.cdf(d1) - L * np.exp(-r * T) * norm.cdf(d2))


def cva_call_option_ind(V0, r_v, sigma_v, T, L, S0, r_s, sigma_s, K, recovery_rate):
    p_default = proba_default(V0, r_v, sigma_v, T, L)
    return (1 - recovery_rate) * p_default * call_option_price(S0, r_s, sigma_s, T, K)
    
    
def integrand(x, y, T, L, S0, r_s, sigma_s, K, rho):
    payoff = (S0 * np.exp((r_s - 0.5 * sigma_s**2) * T + sigma_s * np.sqrt(T) * y)- K)

    density = multivariate_normal.pdf([x, y], mean=[0, 0], cov=[[1, rho], [rho, 1]])

    return payoff * density


def cva_call_option(V0, r_v, sigma_v, T, L, S0, r_s, sigma_s, K, recovery_rate, rho):
    bound1 = (np.log(K / S0) - (r_s - 0.5 * sigma_s**2) * T) / (sigma_s * np.sqrt(T))

    bound2 = (np.log(L / V0) - (r_v - 0.5 * sigma_v**2) * T) / (sigma_v * np.sqrt(T))

    integral = dblquad(integrand, bound1, 40, lambda y: -40, lambda y: bound2, args=(T, L, S0, r_s, sigma_s, K, rho))[0]

    return (1 - recovery_rate) * np.exp(-r_s * T) * integral

#-----------------------
# Black-Cox model functions
#-----------------------


def proba_default_bc(V0, mu_v, sigma_v, T, L):
    m = mu_v - 0.5 * sigma_v**2
    a = np.log(L / V0)
    term1 = norm.cdf((a - m*T) / (sigma_v * np.sqrt(T)))
    term2 = (L / V0)**(2*m / sigma_v**2) * norm.cdf((a + m*T) / (sigma_v * np.sqrt(T)))

    return term1 + term2

def cva_bc_ind(V0, r_v, sigma_v, T, L, S0, r_s, sigma_s, K, recovery_rate):
    p_default = proba_default_bc(V0, r_v, sigma_v, T, L)
    return (1 - recovery_rate) * p_default * call_option_price(S0, r_s, sigma_s, T, K)


def log_gbm_paths(sigma_s, sigma_v, mu_v, rho, V0=100, S0=100, T=1, r_s=0.02, n_paths=100000, n_steps=200, seed=42):
    rng = np.random.default_rng(seed)

    dt = T / n_steps

    # Correlated standard normal shocks
    Z_v = rng.normal(0, 1, size=(n_paths, n_steps))
    Z_int = rng.normal(0, 1, size=(n_paths, n_steps))
    Z_s = rho * Z_v + np.sqrt(1 - rho**2) * Z_int
    #Z_ind = rng.normal(0, 1, size=(n_paths, n_steps))
    #Z_v_ind = rng.normal(0, 1, size=(n_paths, n_steps))

    increments_v = ((mu_v - 0.5 * sigma_v**2) * dt + sigma_v * np.sqrt(dt) * Z_v)

    increments_s = ((r_s - 0.5 * sigma_s**2) * dt + sigma_s * np.sqrt(dt) * Z_s)

    increments_ind = ((r_s - 0.5 * sigma_s**2) * dt + sigma_s * np.sqrt(dt) * Z_int)

    #increments_v_ind = ((mu_v - 0.5 * sigma_v**2) * dt + sigma_v * np.sqrt(dt) * Z_v_ind)
    
    X_v = np.log(V0) + np.cumsum(increments_v, axis=1)
    X_s = np.log(S0) + np.cumsum(increments_s, axis=1)
    X_ind = np.log(S0) + np.cumsum(increments_ind, axis=1)
    #X_v_ind = np.log(V0) + np.cumsum(increments_v_ind, axis=1)

    X_v = np.insert(X_v, 0, np.log(V0), axis=1)
    X_s = np.insert(X_s, 0, np.log(S0), axis=1)
    X_ind = np.insert(X_ind, 0, np.log(S0), axis=1)
    #X_v_ind = np.insert(X_v_ind, 0, np.log(V0), axis=1)

    return X_v, X_s, X_ind #, X_v_ind

def survival_probability_matrix(paths, dt, sigma_v, L):
    X1 = paths[:, :-1]
    X2 = paths[:, 1:]


    log_L = np.log(L)

    q = np.zeros_like(X1)

    no_point_under_barrier = (X1 > log_L) & (X2 > log_L)
    q[no_point_under_barrier] = 1 - np.exp(-2*(X1[no_point_under_barrier] - log_L)*(X2[no_point_under_barrier] - log_L)/((sigma_v**2)*dt))

    return q



def black_cox_mc(L_over_V0, S0_over_K, sigma_s, sigma_v, mu_v, rho, V0=100, S0=100, T=1, r_s=0.02, n_paths=100000, n_steps=200, recovery_rate=0.4, seed=42):
    L = V0 * L_over_V0
    K = S0 / S0_over_K
    dt = T / n_steps

    X_v, X_s, _ = log_gbm_paths(sigma_s, sigma_v, mu_v, rho, V0, S0, T, r_s, n_paths, n_steps, seed)

    Q = np.prod(survival_probability_matrix(X_v, dt, sigma_v, L), axis=1)
    call_payoffs = np.maximum(np.exp(X_s[:, -1]) - K, 0)
    payoffs = call_payoffs * (1-Q)

    discounted_losses = (1 - recovery_rate)* (np.exp(-r_s * T)) * payoffs

    cva, std_error = np.mean(discounted_losses), np.std(discounted_losses, ddof=1) / np.sqrt(n_paths)

    return cva, std_error

def black_cox_mc_control(L_over_V0, S0_over_K, sigma_s, sigma_v, mu_v, rho, V0=100, S0=100, T=1, r_s=0.02, n_paths=100000, n_steps=200, recovery_rate=0.4, seed=42):
    L = V0 * L_over_V0
    K = S0 / S0_over_K
    dt = T / n_steps

    X_v, X_s, X_ind = log_gbm_paths(sigma_s, sigma_v, mu_v, rho, V0, S0, T, r_s, n_paths, n_steps, seed)

    Q = np.prod(survival_probability_matrix(X_v, dt, sigma_v, L), axis=1)
    #Q_ind = np.prod(survival_probability_matrix(X_v_ind, dt, sigma_v, L), axis=1)

    call_payoffs = np.maximum(np.exp(X_s[:, -1]) - K, 0)

    call_payoffs_ind = np.maximum(np.exp(X_ind[:, -1]) - K, 0)

    payoffs = call_payoffs * (1-Q)

    #payoffs_ind = call_payoffs_ind * (1-Q_ind)

    Y = (1 - recovery_rate)* (np.exp(-r_s * T)) * payoffs

    control_variate = (1 - recovery_rate) * (np.exp(-r_s * T)) * call_payoffs_ind * (1-Q)

    if np.var(control_variate) <= 1e-10:
        discounted_losses = Y
    else:
        c = - np.cov(Y, control_variate, ddof=1)[0, 1] / np.var(control_variate, ddof=1)
        discounted_losses = Y + c * (control_variate - cva_bc_ind(V0, r_v=mu_v, sigma_v=sigma_v, T=T, L=L, S0=S0, r_s=r_s, sigma_s=sigma_s, K=K, recovery_rate=recovery_rate))
    
    cva, std_error = np.mean(discounted_losses), np.std(discounted_losses, ddof=1) / np.sqrt(n_paths)

    return cva, std_error
