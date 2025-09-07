#%% Definition of useful functions for the whole project

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import math
from scipy.misc import derivative
from random import *


np.random.seed(42)

def box_muller():
    u1 = np.random.rand()
    u2 = np.random.rand()
    x = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    return(x)

def simu_normal(n):
    simu = np.zeros(n)
    for k in range(n):
        simu[k] = box_muller()
    return(simu)

def densite_normal(x):
    d = (1/(np.sqrt(2*np.pi))) * np.exp(-0.5 * x**2)
    return(d)

#Theorical Price using Black-Scholes Model
def Black_Scholes(option_type, s, k, t, r, sigma):
    d1 = (np.log(k/s) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        return (s*norm.cdf(d1) - (k * np.exp(-r*t)* norm.cdf(d2)))
    else:
        return ((k * np.exp(-r*t)* norm.cdf(-d2)) - s*norm.cdf(-d1))
        
def mult_brownian_paths(t, steps, S_0, number_paths, mu, sigma):  #number_years = 1, 2 or 3... and trading_days = 252 generally
    PATHS = np.zeros((number_paths, steps)) # List of the different prices paths
    delta_t = t/steps
    for i in range(number_paths):
        PATHS[i, 0] = S_0
        for j in range(1, steps):
            eps = box_muller()
            PATHS[i,j] = PATHS[i,j-1] * np.exp(((mu - 0.5 * sigma**2) * delta_t) + sigma * eps * np.sqrt(delta_t))
    return PATHS

def mult_brownian_paths_antithetic(t, steps, S_0, number_paths, mu, sigma):  #number_years = 1, 2 or 3... and trading_days = 252 generally
    PATHS_1 = np.zeros((number_paths, steps)) # List of the different prices paths
    PATHS_2 = np.zeros((number_paths, steps))
    delta_t = t/steps
    for i in range(number_paths):
        PATHS_1[i, 0] = S_0
        PATHS_2[i, 0] = S_0
        for j in range(1, steps):
            eps = box_muller()
            PATHS_1[i,j] = PATHS_1[i,j-1] * np.exp(((mu - 0.5 * sigma**2) * delta_t) + sigma * eps * np.sqrt(delta_t))
            PATHS_2[i,j] = PATHS_2[i,j-1] * np.exp(((mu - 0.5 * sigma**2) * delta_t) - sigma * eps * np.sqrt(delta_t))
    return PATHS_1, PATHS_2


def plot_paths(paths):
    plt.figure()
    plt.xlabel("Steps")
    plt.ylabel("Simulated Stock Price")
    for k in range(paths.shape[0]):
        plt.plot(paths[k])
    plt.savefig("Simulated Stock Price")
    plt.show()

def delta_european(s, k, t, r, sigma, option_type="call"):
    d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    return delta


def geometric_asian_option_price(S_0, K, r, sigma, T, option_type, steps):
    sigma_adj = sigma * np.sqrt((2 * steps + 1) / (6 * (steps + 1)))
    mu_adj = 0.5 * (r - 0.5 * sigma ** 2) * (steps + 1) / (steps + 2) + 0.5 * sigma_adj ** 2
    d1 = (np.log(S_0 / K) + (mu_adj + 0.5 * sigma_adj ** 2) * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)
    
    if option_type == "call":
        price = np.exp(-r * T) * (S_0 * np.exp(mu_adj * T) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - S_0 * np.exp(mu_adj * T) * norm.cdf(-d1))
    
    return price

def conv_speed(list, precision):
    limit = list[-1]
    sup = limit * (1+precision)
    inf = limit*(1-precision)
    for i in range(len(list)-1, -1, -1):
        if list[i]>sup or list[i]<inf:
            return 1/i
    
    return -1 
    