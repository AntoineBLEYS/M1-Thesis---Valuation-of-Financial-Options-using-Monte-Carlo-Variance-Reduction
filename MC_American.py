#%% Settings of the simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
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
NUMBER_PATHS = 10**4                                                              #Number of Simulations
K = S_0                                                                             #Strike Price (ATM Option)
T = 60/365 
STEPS = 50                                                                         #Number of year until expiration of the contract                                             #Number of trading day until expiration
OPTION_TYPE = "call"

print("The parameters of the simulation are:")
print("")
print(f"Stock {OPTION_TYPE} option on {STOCK}")
print("")
print(f"Initial Price = {round(S_0, 5)}")
print(f"Annualized Volatility = {round(SIGMA, 5)}")
print(f"Risk free rate = {R}")
print(f"Number of steps = {STEPS}")
print(f"Strike Price = {K}")
print("")

#%% Pricing American Option with Classic MC
# Taken from : https://medium.com/@ptlabadie/pricing-american-options-in-python-8e357221d2a9
start = time.time()

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)

def Longstaff_Schwartz_American(paths, option_type, k, r, t, sigma):
    n = paths.shape[0]
    Payoffs = np.zeros_like(paths)                                                      #Defining a Payoff Matrix

    european_price_exact = Black_Scholes(option_type, paths[0,0], k, t, r, sigma)
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = np.maximum(paths-k,0)
    else:
        Payoffs = np.maximum(k-paths,0)

    Actualized_Payoffs = np.zeros_like(Payoffs)

    D = Payoffs.shape[1] - 1

    for t in range(1,D):

        in_the_money =paths[:,t] < k

        X = (paths[in_the_money,t])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = Payoffs[in_the_money, t-1]  * np.exp(-r*t/D)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[:,t])
        continuations[in_the_money] = conditional_exp

        Payoffs[:,t] = np.where(continuations> Payoffs[:,t], 0, Payoffs[:,t])

        exercised_early = continuations < Payoffs[:,t]
        Payoffs[:,0:t][exercised_early, :] = 0
        Actualized_Payoffs[:,t-1] = Payoffs[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs[:,D-1] = Payoffs[:,D-1]* np.exp(-r * t/D)
    
    final_cfs = np.zeros((Actualized_Payoffs.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(Actualized_Payoffs[i,:])
    PRICE = np.mean(final_cfs)
    SE = final_cfs.std()/np.sqrt(n)
    print("Using the Longstaff-Schwartz Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)


Longstaff_Schwartz_American(PATHS, OPTION_TYPE, K, R, T, SIGMA)
print(time.time() - start)
# %% Pricing American Options using antithetic MonteCarlo
start = time.time()
PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_MC_American(paths_1, paths_2, option_type, k, r, t):
    n = paths_1.shape[0]                                                     #Defining a Payoff Matrix

    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs1 = np.maximum(paths_1-k,0)
        Payoffs2 = np.maximum(paths_2-k,0)
    else:
        Payoffs1 = np.maximum(k-paths_1,0)
        Payoffs2 = np.maximum(k-paths_2,0)

    Actualized_Payoffs1 = np.zeros_like(Payoffs1)
    Actualized_Payoffs2 = np.zeros_like(Payoffs2)

    D = Payoffs1.shape[1] - 1

    for t in range(1,D):

        in_the_money_1 = paths_1[:,t] < k
        in_the_money_2 = paths_2[:,t] < k

        X_1 = (paths_1[in_the_money_1,t])
        X_2 = (paths_2[in_the_money_2,t])
        X2_1 = X_1*X_1
        X2_2 = X_2*X_2
        Xs_1 = np.column_stack([X_1,X2_1])
        Xs_2 = np.column_stack([X_2,X2_2])
        Y_1 = Payoffs1[in_the_money_1, t-1]  * np.exp(-r*t/D)
        Y_2 = Payoffs2[in_the_money_2, t-1]  * np.exp(-r*t/D)
        model_sklearn = LinearRegression()
        model_1 = model_sklearn.fit(Xs_1, Y_1)
        model_2 = model_sklearn.fit(Xs_2, Y_2)
        conditional_exp_1 = model_1.predict(Xs_1)
        conditional_exp_2 = model_2.predict(Xs_2)
        continuations_1 = np.zeros_like(paths_1[:,t])
        continuations_2 = np.zeros_like(paths_2[:,t])
        continuations_1[in_the_money_1] = conditional_exp_1
        continuations_2[in_the_money_2] = conditional_exp_2

        Payoffs1[:,t] = np.where(continuations_1 > Payoffs1[:,t], 0, Payoffs1[:,t])
        Payoffs2[:,t] = np.where(continuations_2 > Payoffs2[:,t], 0, Payoffs2[:,t])

        exercised_early1 = continuations_1 < Payoffs1[:,t]
        exercised_early2 = continuations_2 < Payoffs2[:,t]

        Payoffs1[:,0:t][exercised_early1, :] = 0
        Payoffs2[:,0:t][exercised_early2, :] = 0

        Actualized_Payoffs1[:,t-1] = Payoffs1[:,t-1]* np.exp(-r * T)
        Actualized_Payoffs2[:,t-1] = Payoffs2[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs1[:,D-1] = Payoffs1[:,D-1]* np.exp(-r * t/D)
    Actualized_Payoffs2[:,D-1] = Payoffs2[:,D-1]* np.exp(-r * t/D)
    
    final_cfs1 = np.zeros((Actualized_Payoffs1.shape[0], 1), dtype=float)
    final_cfs2 = np.zeros((Actualized_Payoffs2.shape[0], 1), dtype=float)    
    for i,row in enumerate(final_cfs1):
        final_cfs1[i] = sum(Actualized_Payoffs1[i,:])
    for j,row in enumerate(final_cfs2):
        final_cfs2[j] = sum(Actualized_Payoffs2[j,:])

    final_cfs_antithetic = (final_cfs1 + final_cfs2)/2
    PRICE = np.mean(final_cfs_antithetic)
    SE = final_cfs_antithetic.std()/np.sqrt(n)
    print("Using the Antithetic Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Antithetic_MC_American(PATHS_1, PATHS_2, OPTION_TYPE, K, R, T)
print(time.time() - start)
#%% Pricing American Options using European Price Control Variable

start = time.time()
import scipy.stats as stats

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)


def Longstaff_Schwartz_American_Control_Variate(paths, option_type, k, r, T, sigma):
    n = paths.shape[0]
    Payoffs = np.zeros_like(paths)                                                      #Defining a Payoff Matrix

    
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = np.maximum(paths-k,0)
    else:
        Payoffs = np.maximum(k-paths,0)

    Actualized_Payoffs = np.zeros_like(Payoffs)

    D = Payoffs.shape[1] - 1

    for t in range(1,D):

        in_the_money =paths[:,t] < k

        X = (paths[in_the_money,t])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = Payoffs[in_the_money, t-1]  * np.exp(-r*t/D)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[:,t])
        continuations[in_the_money] = conditional_exp

        Payoffs[:,t] = np.where(continuations> Payoffs[:,t], 0, Payoffs[:,t])

        exercised_early = continuations < Payoffs[:,t]
        Payoffs[:,0:t][exercised_early, :] = 0
        Actualized_Payoffs[:,t-1] = Payoffs[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs[:,D-1] = Payoffs[:,D-1]* np.exp(-r * t/D)
    
    final_cfs = np.zeros((Actualized_Payoffs.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(Actualized_Payoffs[i,:])
    
    # Control variate adjustment
    european_option_prices = Black_Scholes(option_type, paths[:,0], k, T, r, sigma)
    Z = final_cfs - european_option_prices
    PRICE = np.mean(Z) + np.mean(european_option_prices)
    SE = np.std(Z) / np.sqrt(n)
    
    print("Using the Longstaff-Schwartz Monte Carlo Simulation with Control Variate")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96 * SE, 3)}, {round(PRICE + 1.96 * SE, 3)}]")
    return PRICE, SE

Longstaff_Schwartz_American_Control_Variate(PATHS, OPTION_TYPE, K, R, T, SIGMA)
print(time.time() - start)
# %% Pricing American Option using Control Variate V2
start = time.time()
import scipy.stats as stats

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)

def Longstaff_Schwartz_American_Control_Variate(paths, option_type, k, r, T, sigma):
    n = paths.shape[0]
    Payoffs = np.zeros_like(paths)                                                      #Defining a Payoff Matrix

    euro_price_exact = Black_Scholes(option_type, paths[0,0], k, T, r, sigma)
    
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = np.maximum(paths-k,0)
        Payoffs_Euro = np.maximum(paths[:,-1]-k,0)
    else:
        Payoffs = np.maximum(k-paths,0)
        Payoffs_Euro = np.maximum(k-paths[:,-1],0)


    Actualized_Payoffs_Euro = np.exp(-r*T)*Payoffs_Euro
    Actualized_Payoffs = np.zeros_like(Payoffs)

    D = Payoffs.shape[1] - 1

    for t in range(1,D):

        in_the_money =paths[:,t] < k

        X = (paths[in_the_money,t])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = Payoffs[in_the_money, t-1]  * np.exp(-r*t/D)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[:,t])
        continuations[in_the_money] = conditional_exp

        Payoffs[:,t] = np.where(continuations> Payoffs[:,t], 0, Payoffs[:,t])

        exercised_early = continuations < Payoffs[:,t]
        Payoffs[:,0:t][exercised_early, :] = 0
        Actualized_Payoffs[:,t-1] = Payoffs[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs[:,D-1] = Payoffs[:,D-1]* np.exp(-r * t/D)
    
    final_cfs = np.zeros((Actualized_Payoffs.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(Actualized_Payoffs[i,:])
    
    # Control variate adjustment
    Z = final_cfs - Actualized_Payoffs_Euro + euro_price_exact
    PRICE = np.mean(Z)
    SE = np.std(Z) / np.sqrt(n)
    
    print("Using the Longstaff-Schwartz Monte Carlo Simulation with Control Variate and Antithetic")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96 * SE, 3)}, {round(PRICE + 1.96 * SE, 3)}]")
    return PRICE, SE

Longstaff_Schwartz_American_Control_Variate(PATHS, OPTION_TYPE, K, R, T, SIGMA)
print(time.time() - start)
#%%LS Algo with control and antithetics
start = time.time()
PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Longstaff_Schwartz_American_Anti_Control_Variate(paths_1, paths_2, option_type, k, r, T, sigma):
    n = paths_1.shape[0]

    #First Path
    Payoffs_1 = np.zeros_like(paths_1)  
                                        
    euro_price_exact_1 = Black_Scholes(option_type, paths_1[0,0], k, T, r, sigma)
    
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs_1 = np.maximum(paths_1-k,0)
        Payoffs_Euro_1 = np.maximum(paths_1[:,-1]-k,0)
    else:
        Payoffs_1 = np.maximum(k-paths_1,0)
        Payoffs_Euro_1 = np.maximum(k-paths_1[:,-1],0)

    Actualized_Payoffs_Euro_1 = np.exp(-r*T)*Payoffs_Euro_1
    Actualized_Payoffs_1 = np.zeros_like(Payoffs_1)

    D = Payoffs_1.shape[1] - 1

    for t in range(1,D):

        in_the_money_1 =paths_1[:,t] < k

        X = (paths_1[in_the_money_1,t])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = Payoffs_1[in_the_money_1, t-1]  * np.exp(-r*t/D)
        model_sklearn_1 = LinearRegression()
        model_1 = model_sklearn_1.fit(Xs, Y)
        conditional_exp_1 = model_1.predict(Xs)
        continuations_1 = np.zeros_like(paths_1[:,t])
        continuations_1[in_the_money_1] = conditional_exp_1

        Payoffs_1[:,t] = np.where(continuations_1> Payoffs_1[:,t], 0, Payoffs_1[:,t])

        exercised_early_1 = continuations_1 < Payoffs_1[:,t]
        Payoffs_1[:,0:t][exercised_early_1, :] = 0
        Actualized_Payoffs_1[:,t-1] = Payoffs_1[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs_1[:,D-1] = Payoffs_1[:,D-1]* np.exp(-r * t/D)
    
    final_cfs_1 = np.zeros((Actualized_Payoffs_1.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs_1):
        final_cfs_1[i] = sum(Actualized_Payoffs_1[i,:])

    Z1 = final_cfs_1 - Actualized_Payoffs_Euro_1 + euro_price_exact_1


    #Second Path


    Payoffs_2 = np.zeros_like(paths_2)  
                                            

    euro_price_exact_2 = Black_Scholes(option_type, paths_2[0,0], k, T, r, sigma)
    
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs_2 = np.maximum(paths_2-k,0)
        Payoffs_Euro_2 = np.maximum(paths_2[:,-1]-k,0)
    else:
        Payoffs_2 = np.maximum(k-paths_2,0)
        Payoffs_Euro_2 = np.maximum(k-paths_2[:,-1],0)


    Actualized_Payoffs_Euro_2 = np.exp(-r*T)*Payoffs_Euro_2
    Actualized_Payoffs_2 = np.zeros_like(Payoffs_2)

    D = Payoffs_2.shape[1] - 1

    for t in range(1,D):

        in_the_money_2 =paths_2[:,t] < k

        X = (paths_2[in_the_money_2,t])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = Payoffs_2[in_the_money_2, t-1]  * np.exp(-r*t/D)
        model_sklearn_2 = LinearRegression()
        model_2 = model_sklearn_2.fit(Xs, Y)
        conditional_exp_2 = model_2.predict(Xs)
        continuations_2 = np.zeros_like(paths_2[:,t])
        continuations_2[in_the_money_2] = conditional_exp_2

        Payoffs_2[:,t] = np.where(continuations_2> Payoffs_2[:,t], 0, Payoffs_2[:,t])

        exercised_early_2 = continuations_2 < Payoffs_2[:,t]
        Payoffs_2[:,0:t][exercised_early_2, :] = 0
        Actualized_Payoffs_2[:,t-1] = Payoffs_2[:,t-1]* np.exp(-r * T)

    Actualized_Payoffs_2[:,D-1] = Payoffs_2[:,D-1]* np.exp(-r * t/D)
    
    final_cfs_2 = np.zeros((Actualized_Payoffs_2.shape[0], 1), dtype=float)
    for i,row in enumerate(final_cfs_2):
        final_cfs_2[i] = sum(Actualized_Payoffs_2[i,:])
    
    # Control variate adjustment
    Z2 = final_cfs_2 - Actualized_Payoffs_Euro_2 + euro_price_exact_2


    Z = (Z1 + Z2)*0.5
    PRICE = np.mean(Z)
    SE = np.std(Z) / np.sqrt(n)
    
    print("Using the Longstaff-Schwartz Monte Carlo Simulation with Control Variate")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96 * SE, 3)}, {round(PRICE + 1.96 * SE, 3)}]")
    return PRICE, SE

Longstaff_Schwartz_American_Anti_Control_Variate(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T, SIGMA)
print(time.time() - start)
# %% Convergence of different algos

interval = np.arange(24, 2500+4,10)
Price_LS_American = []
Price_Antithetic_American = []
Price_Control_American = []
Price_Anti_Control_American = []

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_LS_American.append(Longstaff_Schwartz_American(PATHS, OPTION_TYPE, K, R, T, SIGMA)[0])

for n in interval:
    PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Antithetic_American.append(Antithetic_MC_American(PATHS_1, PATHS_2, OPTION_TYPE, K, R, T)[0])

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Control_American.append(Longstaff_Schwartz_American_Control_Variate(PATHS, OPTION_TYPE, K, R, T, SIGMA)[0])

for n in interval:
    PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Anti_Control_American.append(Longstaff_Schwartz_American_Anti_Control_Variate(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T, SIGMA)[0])

#%% Plots
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(interval, Price_LS_American, color = 'darkblue', linewidth = 1)
axs[0, 0].set_title('Longstaff MC')
axs[0, 0].grid(True)
axs[0, 0].set_xlabel('Number of Paths')
axs[0, 0].set_ylabel('Option Price')

axs[0, 1].plot(interval, Price_Antithetic_American, color = 'darkred', linewidth = 1)
axs[0, 1].set_title('Antithetic MC')
axs[0, 1].grid(True)
axs[0, 1].set_xlabel('Number of Paths')
axs[0, 1].set_ylabel('Option Price')

axs[1, 0].plot(interval, Price_Control_American, color = 'orange', linewidth = 1)
axs[1, 0].set_title('European Control MC')
axs[1, 0].grid(True)
axs[1, 0].set_xlabel('Number of Paths')
axs[1, 0].set_ylabel('Option Price')

axs[1, 1].plot(interval, Price_Anti_Control_American, color = 'green', linewidth = 1)
axs[1, 1].set_title('Antithetic + European Control MC')
axs[1, 1].grid(True)
axs[1, 1].set_xlabel('Number of Paths')
axs[1, 1].set_ylabel('Option Price')

fig.tight_layout()
plt.savefig("American MC Convergence Speed for each algos")


plt.figure()
plt.plot(interval, Price_LS_American, color = "darkblue", label = 'Classic')
plt.plot(interval, Price_Antithetic_American, color = "darkred", label = 'Antithetic')
plt.plot(interval, Price_Control_American, color = "orange", label = 'European Control')
plt.plot(interval, Price_Anti_Control_American, color = "green", label = 'Antithetic & European Control')
plt.legend()
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.grid()

plt.savefig('American Speed Convergence Comparison')

# %%
