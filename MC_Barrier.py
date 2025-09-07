#%% Settings of the simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from MC_functions import *
import time

np.random.seed(2024)

STOCK = "TTE.PA"                                                                    #TotalEnergie
data = yf.download(STOCK)

stock = data.drop(labels = ['Open', "High", "Low", "Close", "Volume"], axis = 'columns')
stock["Returns"] = stock["Adj Close"] / stock["Adj Close"].shift(1) - 1
stock['Log Return'] = np.log(stock['Adj Close'] / stock['Adj Close'].shift(1))

sigma = stock["Log Return"].std()*np.sqrt(252)                                      #Annualize volatility of Log-Return
print("")


#Parameters of the simulation

R = 0.0390                                                                          #Risk free rate
SIGMA = sigma                                                                       #Volatility
S_0 = stock["Adj Close"].iloc[-1]                                                   #Initial Price
NUMBER_PATHS = 10**5                                                                #Number of Simulations
K = S_0                                                                             #Strike Price (ATM Option)                                                                  #Strike Price (ATM Option)
T = 60/365                                                                          #Number of year until expiration of the contract
STEPS = 50                                                                          #Number of trading day until expiration
OPTION_TYPE = "call"
BARRIER_TYPE = "DO"
B = S_0*np.exp(-R*T)                                                                 #Barrier of the option

print("The parameters of the simulation are:")
print("")
print(f"Stock {BARRIER_TYPE} {OPTION_TYPE} option on {STOCK}")
print("")
print(f"Initial Price = {round(S_0, 5)}")
print(f"Annualized Volatility = {round(SIGMA, 5)}")
print(f"Risk free rate = {R}")
print(f"Number of steps = {STEPS}")
print(f"Strike Price = {K}")
print(f"Barrier = {B}")
print("")


#%% Pricing Barrier Option with Classic MC
start = time.time()
PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)

def Classic_MC_Barrier(paths, option_type, barrier_type, b, k, r, t):
    
    if option_type == "call":
        Payoffs = np.maximum(paths[:,-1]-k,0)
    else:
        Payoffs = np.maximum(k-paths[:,-1],0)
    if barrier_type == "UO":
        Payoffs = Payoffs*((np.all(paths<b,axis=1)).astype(int))
    elif barrier_type == "UI":
        Payoffs = Payoffs*((np.any(paths>=b,axis=1)).astype(int))
    elif barrier_type == "DO":
        Payoffs = Payoffs*((np.all(paths>b,axis=1)).astype(int))
    elif barrier_type == "DI":
        Payoffs = Payoffs*((np.any(paths<=b,axis=1)).astype(int))
    Actualized_Payoffs = np.exp(-r*t)*Payoffs                                      #Actualizing the Payoffs
    PRICE = np.mean(Actualized_Payoffs)                                            #Averaging the payoffs (MC method)
    SE = Actualized_Payoffs.std()/np.sqrt(NUMBER_PATHS)                            #Computing the standard error
    
    print("Using the standard Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Classic_MC_Barrier(PATHS, OPTION_TYPE, BARRIER_TYPE, B, K, R, T)
print(time.time() - start)
#%% Pricing Barrier Options using Antithetic Variable
start = time.time()
PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_MC_Barrier(paths_1, paths_2, option_type, barrier_type, b, k, r, t):
    n = paths_1.shape[0]

    if option_type == "call":
        Payoffs_1 = np.maximum(paths_1[:,-1]-k,0)
        Payoffs_2 = np.maximum(paths_2[:,-1]-k,0)
    else:
        Payoffs_1 = np.maximum(k-paths_1[:,-1],0)
        Payoffs_2 = np.maximum(k-paths_2[:,-1],0)

    if barrier_type == "UO":
        Payoffs_1 = Payoffs_1*((np.all(paths_1<b,axis=1)).astype(int))
        Payoffs_2 = Payoffs_2*((np.all(paths_2<b,axis=1)).astype(int))
    elif barrier_type == "UI":
        Payoffs_1 = Payoffs_1*((np.any(paths_1>=b,axis=1)).astype(int))
        Payoffs_2 = Payoffs_2*((np.any(paths_2>=b,axis=1)).astype(int))
    elif barrier_type == "DO":
        Payoffs_1 = Payoffs_1*((np.all(paths_1>b,axis=1)).astype(int))
        Payoffs_2 = Payoffs_2*((np.all(paths_2>b,axis=1)).astype(int))
    else :
        Payoffs_1 = Payoffs_1*((np.any(paths_1<=b,axis=1)).astype(int))
        Payoffs_2 = Payoffs_2*((np.any(paths_2<=b,axis=1)).astype(int))

    Payoffs = (Payoffs_1 + Payoffs_2)/2
    Actualized_Payoffs = np.exp(-r*t)*Payoffs                                      #Actualizing the Payoffs
    PRICE = np.mean(Actualized_Payoffs)                                            #Averaging the payoffs (MC method)
    SE = Actualized_Payoffs.std()/np.sqrt(n)
    
    print("Using the Antithetic Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Antithetic_MC_Barrier(PATHS_1, PATHS_2, OPTION_TYPE, BARRIER_TYPE, B, K, R, T);
print(time.time() - start)
#%%
interval = np.arange(4, 2500+4,10)
Price_Classic_Barrier = []
Price_Antithetic_Barrier = []

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Classic_Barrier.append(Classic_MC_Barrier(PATHS, OPTION_TYPE, BARRIER_TYPE, B, K, R, R)[0])

for n in interval:
    PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Antithetic_Barrier.append(Antithetic_MC_Barrier(PATHS_1, PATHS_2, OPTION_TYPE, BARRIER_TYPE,B, K, R, T)[0])

#%% Plots
fig, axs = plt.subplots(2, 1)

axs[0].plot(interval, Price_Classic_Barrier, color = 'darkblue', linewidth = 1)
axs[0].set_title('Classic MC')
axs[0].grid(True)
axs[0].set_xlabel('Number of Paths')
axs[0].set_ylabel('Option Price')

axs[1].plot(interval, Price_Antithetic_Barrier, color = 'darkred', linewidth = 1)
axs[1].set_title('Antithetic MC')
axs[1].grid(True)
axs[1].set_xlabel('Number of Paths')
axs[1].set_ylabel('Option Price')


fig.tight_layout()
plt.savefig("Barrier MC Convergence Speed for each algos")


plt.figure()
plt.plot(interval, Price_Classic_Barrier, color = "darkblue", label = 'Classic')
plt.plot(interval, Price_Antithetic_Barrier, color = "darkred", label = 'Antithetic')
plt.legend()
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.grid()

plt.savefig('Barrier Speed Convergence Comparison')

# %%
