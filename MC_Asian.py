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
NUMBER_PATHS = 10**3                                                              #Number of Simulations                                                                             #Strike Price (ATM Option)
T = 60/365                                                                          #Number of year until expiration of the contract
STEPS = 50  
K = S_0                                           #Number of trading day until expiration
OPTION_TYPE = "call"

print("The parameters of the simulation are:")
print("")
print(f"Stock {OPTION_TYPE} option on {STOCK}")
print("")
print(f"Initial Price = {round(S_0, 5)}")
print(f"Annualized Volatility = {round(SIGMA, 5)}")
print(f"Risk free rate = {R}")
print(f"Number of steps = {STEPS}")
print("")

#%% Pricing Asian Option with Classic MC

start = time.time()

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)   #Generating the differents paths

def Classic_MC_Asian(paths, k, option_type, r, t):                                                  #Vector of strikes for each path simulation
    if option_type == "call":
        Payoffs = np.maximum(np.mean(paths, axis = 1) - k,0)
    else:
        Payoffs = np.maximum(k - np.mean(paths, axis = 1),0)
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

Classic_MC_Asian(PATHS, K, OPTION_TYPE, R, T)

print(time.time() - start)
#%% Pricing Asian Options using Antithetic Variable
start = time.time()
PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_MC_Asian(paths_1, paths_2, k, option_type, r, t):
    n = paths_1.shape[0]                                                   #Vector of strikes for each path_2 simulation
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        Payoffs = (np.maximum(np.mean(paths_1, axis = 1) - k,0) + np.maximum(np.mean(paths_2, axis = 1)-k,0))/2
    else:
        Payoffs = (np.maximum(k - np.mean(paths_1, axis = 1) ,0) + np.maximum(k-np.mean(paths_2, axis = 1),0))/2

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

Antithetic_MC_Asian(PATHS_1, PATHS_2, K,OPTION_TYPE, R, T);
print(time.time() - start)
#%% Pricing Asian Options using Vorst Control Variable

start = time.time()

PATHS = mult_brownian_paths(T, STEPS, S_0, NUMBER_PATHS, R, SIGMA)   #Generating the differents paths
# Geometric Asian option price
geom_price_exact = geometric_asian_option_price(S_0, K, R, SIGMA, T, OPTION_TYPE, STEPS)
print(f"Geometric Asian option price (exact) = {round(geom_price_exact, 5)}€")

# Function for Monte Carlo pricing with control variate
def Vorst_Control_MC_Asian(paths, k, option_type, r, t):
    if option_type == "call":
        arith_payoffs = np.maximum(np.mean(paths[:, 1:], axis=1) - k, 0)
        geom_means = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        geom_payoffs = np.maximum(geom_means - k, 0)
    else:
        arith_payoffs = np.maximum(k - np.mean(paths[:, 1:], axis=1), 0)
        geom_means = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        geom_payoffs = np.maximum(k - geom_means, 0)
    
    actualized_arith = np.exp(-r * t) * arith_payoffs
    actualized_geom = np.exp(-r * t) * geom_payoffs
    Actualized_Payoffs = actualized_arith - actualized_geom + geom_price_exact

    PRICE = np.mean(Actualized_Payoffs)                                     #Averaging the payoffs on the last date provide the option price
    SE = Actualized_Payoffs.std()/np.sqrt(NUMBER_PATHS)

    
    print("Using the Monte Carlo Simulation with Control Variate")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96 * SE, 5)}, {round(PRICE + 1.96 * SE, 5)}]")
    return PRICE, SE

Vorst_Control_MC_Asian(PATHS, K, OPTION_TYPE, R, T)

print(time.time() - start)


#%% Pricing Asians Options using Antithetic and Control Variables

start = time.time()

PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, NUMBER_PATHS//2, R, SIGMA)

def Antithetic_Control_MC_Asian(paths_1, paths_2, option_type, k, r, t):
    n = paths_1.shape[0]

    geom_price_exact = geometric_asian_option_price(S_0, K, R, SIGMA, T, OPTION_TYPE, STEPS)
    
    if option_type == "call":                                                           #Computing the Payoff at each nodes of the simulation
        arith_payoffs1 = np.maximum(np.mean(paths_1[:, 1:], axis=1) - k, 0)
        geom_means1 = np.exp(np.mean(np.log(paths_1[:, 1:]), axis=1))
        geom_payoffs1 = np.maximum(geom_means1 - k, 0)

        arith_payoffs2 = np.maximum(np.mean(paths_2[:, 1:], axis=1) - k, 0)
        geom_means2 = np.exp(np.mean(np.log(paths_2[:, 1:]), axis=1))
        geom_payoffs2 = np.maximum(geom_means2 - k, 0)
    else:
        arith_payoffs1 = np.maximum(k - np.mean(paths_1[:, 1:], axis=1), 0)
        geom_means1 = np.exp(np.mean(np.log(paths_1[:, 1:]), axis=1))
        geom_payoffs1 = np.maximum(k - geom_means1, 0)

        arith_payoffs2 = np.maximum(k - np.mean(paths_2[:, 1:], axis=1), 0)
        geom_means2 = np.exp(np.mean(np.log(paths_2[:, 1:]), axis=1))
        geom_payoffs2 = np.maximum(k - geom_means2, 0)

    actualized_arith1 = np.exp(-r * t) * arith_payoffs1
    actualized_geom1 = np.exp(-r * t) * geom_payoffs1
    Actualized_Payoffs1 = actualized_arith1 - actualized_geom1 + geom_price_exact

    actualized_arith2 = np.exp(-r * t) * arith_payoffs2
    actualized_geom2 = np.exp(-r * t) * geom_payoffs2
    Actualized_Payoffs2 = actualized_arith2 - actualized_geom2 + geom_price_exact

    Actualized_Controled_Payoffs = (Actualized_Payoffs1 + Actualized_Payoffs2)/2

    PRICE = np.mean(Actualized_Controled_Payoffs)
    SE = Actualized_Controled_Payoffs.std()/np.sqrt(n)

    print("Using Antithetic and Vorst control variable Monte Carlo Simulation")
    print(f"Price of the option is = {round(PRICE, 5)}€")
    print(f"Standard Error for the price = {round(SE, 5)}")
    print("")
    print("The 95% confidence interval for the true value of the price :")
    print(f"[{round(PRICE - 1.96*SE,3)}, {round(PRICE + 1.96*SE, 3)}]")
    return(PRICE, SE)

Antithetic_Control_MC_Asian(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T)
print(time.time() - start)



#%% Defining delta function for asian options
def Classic_MC_Asian2(paths, k, option_type, r, t):                                                  #Vector of strikes for each path simulation
    if option_type == "call":
        Payoffs = np.maximum(np.mean(paths, axis = 1) - k,0)
    else:
        Payoffs = np.maximum(k - np.mean(paths, axis = 1),0)
    Actualized_Payoffs = np.exp(-r*t)*Payoffs                                      
    PRICE = np.mean(Actualized_Payoffs)                                 
    return(PRICE)

def only_price_asian(x):
    PATHS = mult_brownian_paths(T, STEPS, x, NUMBER_PATHS, R, SIGMA)
    p = Classic_MC_Asian2(PATHS, K, OPTION_TYPE, R, T);
    return p;

def delta_asian(x):
    return derivative(only_price_asian,x)

print(only_price_asian(S_0))
print(delta_asian(S_0))

x = np.linspace(50, 80 , 100)
y = np.zeros(100)
for k in range(100):
    y[k] = delta_asian(x[k])
plt.plot(x,y)

#%% Convergence Classic

interval = np.arange(4, 2500+4,10)
Price_Classic_Asian = []
Price_Antithetic_Asian = []
Price_Vorst_Asian = []
Price_Antithetic_Vorst_Asian = []

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Classic_Asian.append(Classic_MC_Asian(PATHS, K, OPTION_TYPE, R, T)[0])

for n in interval:
    PATHS_1, PATHS_2 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Antithetic_Asian.append(Antithetic_MC_Asian(PATHS_1, PATHS_2, K, OPTION_TYPE, R, T)[0])

for n in interval:
    PATHS = mult_brownian_paths(T, STEPS, S_0, n, R, SIGMA)
    Price_Vorst_Asian.append(Vorst_Control_MC_Asian(PATHS, K, OPTION_TYPE, R, T)[0])

for n in interval:
    PATHS_3, PATHS_4 = mult_brownian_paths_antithetic(T, STEPS, S_0, n//2, R, SIGMA)
    Price_Antithetic_Vorst_Asian.append(Antithetic_Control_MC_Asian(PATHS_3, PATHS_4, OPTION_TYPE, K, R, T)[0])

#%% MPlots


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(interval, Price_Classic_Asian, color = 'darkblue', linewidth = 1)
axs[0, 0].set_title('Classic MC')
axs[0, 0].grid(True)
axs[0, 0].set_xlabel('Number of Paths')
axs[0, 0].set_ylabel('Option Price')

axs[0, 1].plot(interval, Price_Antithetic_Asian, color = 'darkred', linewidth = 1)
axs[0, 1].set_title('Antithetic MC')
axs[0, 1].grid(True)
axs[0, 1].set_xlabel('Number of Paths')
axs[0, 1].set_ylabel('Option Price')

axs[1, 0].plot(interval, Price_Vorst_Asian, color = 'orange', linewidth = 1)
axs[1, 0].set_title('Vorst Control MC')
axs[1, 0].grid(True)
axs[1, 0].set_xlabel('Number of Paths')
axs[1, 0].set_ylabel('Option Price')

axs[1, 1].plot(interval, Price_Antithetic_Vorst_Asian, color = 'green', linewidth = 1)
axs[1, 1].set_title('Antithetic + Vorst Control MC')
axs[1, 1].grid(True)
axs[1, 1].set_xlabel('Number of Paths')
axs[1, 1].set_ylabel('Option Price')

fig.tight_layout()
plt.savefig("Asian MC Convergence Speed for each algos")


plt.figure()
plt.plot(interval, Price_Classic_Asian, color = "darkblue", label = 'Classic')
plt.plot(interval, Price_Antithetic_Asian, color = "darkred", label = 'Antithetic')
plt.plot(interval, Price_Vorst_Asian, color = "orange", label = 'Vorst Control')
plt.plot(interval, Price_Antithetic_Vorst_Asian, color = "green", label = 'Antithetic & Vorst Control')
plt.legend()
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.grid()

plt.savefig('Asian Speed Convergence Comparison')

# %%

plt.figure()
plt.plot(interval, Price_Vorst_Asian, color = "orange", label = 'Vorst Control')
plt.plot(interval, Price_Antithetic_Vorst_Asian, color = "green", label = 'Antithetic & Vorst Control')
plt.legend()
plt.xlabel("Number of Paths")
plt.ylabel("Option Price")
plt.grid()

plt.savefig('Asian Speed Convergence Comparison Two Bests algorithms')
# %%
