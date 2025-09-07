#%% Settings of the simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import math
import time
from MC_functions import *


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
NUMBER_PATHS = 10**5                                                               #Number of Simulations
K = S_0                                                                             #Strike Price (ATM Option)
T = 60/365                                                                          #Number of year until expiration of the contract
STEPS = 50                                                                       #Number of trading day until expiration
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

print("With those settings, the theorical price using")
print(f"BS Model is : Option_Price = {round(Black_Scholes(OPTION_TYPE, S_0, K, T, R, SIGMA), 3)}€")

#%% PATHS Simulation 

#PATHS = mult_brownian_paths(T, STEPS, S_0, 100, R, SIGMA)
#plot_paths(PATHS)

#%% Pricing European Option with Classic MC

start = time.time()

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)

def Classic_MC_European(paths, option_type, k, r, t, steps):
    dt = t/steps
    n = paths.shape[0]
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = np.maximum(paths[:,-1]-k,0)
    else:
        Payoffs = np.maximum(k-paths[:,-1],0)

    Actualized_Payoffs = np.exp(-r*t)*Payoffs                                           #Actualizing the Payoffs
    PRICE = np.mean(Actualized_Payoffs)                                            #Averaging the payoffs on the last date provide the option price
    SE = Actualized_Payoffs.std()/np.sqrt(n)                                 #Computing the standard error
    

    print("Using the standard Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Classic_MC_European(PATHS, OPTION_TYPE, K, R, T, STEPS);

print(time.time() - start)
#%% Pricing European Options using Antithetic Variable

start = time.time()
PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_MC_European(paths_1, paths_2, option_type, k, r, t, steps):
    dt = t/steps
    n = paths_1.shape[0]

    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = (np.maximum(paths_1[:,-1]-k,0) + np.maximum(paths_2[:,-1]-k,0))/2
    else:
        Payoffs = (np.maximum(k-paths_1[:,-1],0) + np.maximum(k-paths_2[:,-1],0))/2

    Actualized_Payoffs = np.exp(-r*t)*Payoffs                                          #Actualizing the Payoffs
    PRICE = np.mean(Actualized_Payoffs)                                            #Averaging the payoffs on the last date provide the option price
    SE = Actualized_Payoffs.std()/np.sqrt(n)                                 #Computing the standard error
    

    print("Using the Antithetic Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Antithetic_MC_European(PATHS_1, PATHS_2, OPTION_TYPE, K, R, T, STEPS);
print(time.time() - start)

#%% Pricing European Options using Delta Hedge Control Variable
start = time.time()
PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)

def Delta_Control_MC_European(paths, option_type, k, r, t, steps):
    dt = t/steps
    n = paths.shape[0]

    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = np.maximum(paths[:,-1]-k,0)
    else:
        Payoffs = np.maximum(k-paths[:,-1],0)

    delta_matrix = delta_european(paths[:, :-1], k, np.linspace(t, 0, steps)[:-1], r, SIGMA, option_type)
    control_matrix = delta_matrix*(paths[:,1:] - paths[:,:-1]*np.exp(r*dt))*np.exp(r*np.linspace(t, 0, steps-1))
    control_variate_total = np.sum(control_matrix, axis=1)
    Controled_Payoffs = Payoffs - control_variate_total

    #control_variate_total = np.cumsum(delta_matrix*(paths[:,1:] - paths[:,:-1]*np.exp(r*dt))*np.exp(r*np.linspace(t, 0, steps-1)), axis=1)
    #Controled_Payoffs = Payoffs - control_variate_total[:,-1]

    Actualized_Controled_Payoffs = np.exp(-r*t)*Controled_Payoffs
    PRICE = np.mean(Actualized_Controled_Payoffs)
    SE = Actualized_Controled_Payoffs.std()/np.sqrt(n)

    print("Using the Delta hedging control variable Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Delta_Control_MC_European(PATHS, OPTION_TYPE, K, R, T, STEPS)
print(time.time() - start)

#%% Pricing European Options using Antithetic and Control Variables
start = time.time()

PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_Control_MC_European(paths_1, paths_2, option_type, k, r, t, steps):
    dt = t/steps
    n = paths_1.shape[0]

    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = (np.maximum(paths_1[:,-1]-k,0) + np.maximum(paths_2[:,-1]-k,0))/2
    else:
        Payoffs = (np.maximum(k-paths_1[:,-1],0) + np.maximum(k-paths_2[:,-1],0))/2


    delta_matrix_1 = delta_european(paths_1[:, :-1], k, np.linspace(t, 0, steps)[:-1], r, SIGMA, option_type)
    control_matrix_1 = delta_matrix_1*(paths_1[:,1:] - paths_1[:,:-1]*np.exp(r*dt))*np.exp(r*np.linspace(t, 0, steps-1))
    control_variate_total_1 = np.sum(control_matrix_1, axis=1)

    delta_matrix_2 = delta_european(paths_2[:, :-1], k, np.linspace(t, 0, steps)[:-1], r, SIGMA, option_type)
    control_matrix_2 = delta_matrix_2*(paths_2[:,1:] - paths_2[:,:-1]*np.exp(r*dt))*np.exp(r*np.linspace(t, 0, steps-1))
    control_variate_total_2 = np.sum(control_matrix_2, axis=1)
    
    control_variate_total = (control_variate_total_1 + control_variate_total_2)/2

    Controled_Payoffs = (Payoffs - control_variate_total)
    Actualized_Controled_Payoffs = np.exp(-r*t)*Controled_Payoffs
    PRICE = np.mean(Actualized_Controled_Payoffs)
    SE = Actualized_Controled_Payoffs.std()/np.sqrt(n)

    print("Using Antithetic and Delta hedging control variable Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Antithetic_Control_MC_European(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T, STEPS)
print(time.time() - start)

#%% Convergence of different algos

interval = np.arange(4, 2500+4,10)
Price_Classic_European = []
Price_Antithetic_European = []
Price_Delta_European = []
Price_Anti_Delta_European = []

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Classic_European.append(Classic_MC_European(PATHS, OPTION_TYPE, K, R, T, STEPS)[0])

for n in interval:
    PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Antithetic_European.append(Antithetic_MC_European(PATHS_1, PATHS_2, OPTION_TYPE, K, R, T, STEPS)[0])

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Delta_European.append(Delta_Control_MC_European(PATHS, OPTION_TYPE, K, R, T, STEPS)[0])

for n in interval:
    PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Anti_Delta_European.append(Antithetic_Control_MC_European(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T, STEPS)[0])

#%% Plots
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(interval, Price_Classic_European, color = 'darkblue', linewidth = 1)
axs[0, 0].set_title('Classic MC')
axs[0, 0].grid(True)
axs[0, 0].set_xlabel('Number of Paths')
axs[0, 0].set_ylabel('Option Price')

axs[0, 1].plot(interval, Price_Antithetic_European, color = 'darkred', linewidth = 1)
axs[0, 1].set_title('Antithetic MC')
axs[0, 1].grid(True)
axs[0, 1].set_xlabel('Number of Paths')
axs[0, 1].set_ylabel('Option Price')

axs[1, 0].plot(interval, Price_Delta_European, color = 'orange', linewidth = 1)
axs[1, 0].set_title('Delta Control MC')
axs[1, 0].grid(True)
axs[1, 0].set_xlabel('Number of Paths')
axs[1, 0].set_ylabel('Option Price')

axs[1, 1].plot(interval, Price_Anti_Delta_European, color = 'green', linewidth = 1)
axs[1, 1].set_title('Antithetic + Delta Control MC')
axs[1, 1].grid(True)
axs[1, 1].set_xlabel('Number of Paths')
axs[1, 1].set_ylabel('Option Price')

fig.tight_layout()
plt.savefig("European MC Convergence Speed for each algos")


plt.figure()
plt.plot(interval, Price_Classic_European, color = "darkblue", label = 'Classic')
plt.plot(interval, Price_Antithetic_European, color = "darkred", label = 'Antithetic')
plt.plot(interval, Price_Delta_European, color = "orange", label = 'Delta Control')
plt.plot(interval, Price_Anti_Delta_European, color = "green", label = 'Antithetic & Delta Control')
plt.legend()
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.grid()

plt.savefig('European Speed Convergence Comparison')


# %%
def conv_speed(list, precision):
    limit = list[-1]
    sup = limit * (1+precision)
    inf = limit*(1-precision)
    for i in range(len(list)-1, 0, -1):
        if list[i]>sup or list[i]<inf:
            return 1/i
    
    return -1              #If all value in interval

print(conv_speed(Price_Classic_European, 0.025))
print(conv_speed(Price_Antithetic_European, 0.025))
print(conv_speed(Price_Delta_European, 0.025))
print(conv_speed(Price_Anti_Delta_European, 0.025))
# %%
