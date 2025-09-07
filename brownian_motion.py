"""
@authors: Antoine BLS, Anthony JCB

Link for the bibliography
https://1drv.ms/w/s!AuO2mMwUJ19JgfgnS-v-Km827acK0w?e=lc6GyO

Link for the Mémoire
https://1drv.ms/w/s!AuO2mMwUJ19JgowF8krZAuqoVNT-aw?e=sv7fQC

Link for google meet
https://meet.google.com/dii-pobo-cru?pli=1
"""

#%% Simulation of a normal sampling with Box Muller (computation takes few seconds)

import numpy as np
import matplotlib.pyplot as plt

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

x = np.linspace(-5, 5, 10**3)
y = densite_normal(x)

plt.figure()
plt.hist(simu_normal(10**6), color = 'darkblue', bins = 10**2, density = True)
plt.plot(x,y, color = "red")

#%% Getting the data and average return and volatility

import yfinance as yf

data1 = yf.download("TTE.PA", start = "2010-01-01") #LVMH
#data1 = data.tail(NUMBER_DAYS)
stock = data.drop(labels = ['Open', "High", "Low", "Close", "Volume"], axis = 'columns')
stock["Returns"] = stock["Adj Close"] / stock["Adj Close"].shift(1) - 1
stock['Log Return'] = np.log(stock['Adj Close'] / stock['Adj Close'].shift(1))
print(data1)

mu = (stock["Returns"].mean() + 1)**252 - 1
print(f"Average Return of {round(100 * mu, 5)} %")

sigma = stock["Log Return"].std()*np.sqrt(252)
print(f"Average Standard Deviation of {round(sigma, 5)}")

# Computing the mean return and mean volatility



#%% Simulating a brownian trajectory

def brownian_traj(number_days, S_0):  #number_years = 1, 2 or 3... and trading_days = 252 generally
   
    simu = np.zeros(number_days)
    delta_t = 1/(number_days)
    simu[0] = S_0
    S_new = S_0
    for k in range(1, number_days):
        eps = np.random.normal(0, 1 ,1)
        S_new = S_new * np.exp(((mu - 0.5 * sigma**2) * delta_t) + sigma * eps * np.sqrt(delta_t))
        simu[k] = S_new
    return simu


S_0 = data1["Adj Close"][-1]
plt.plot(brownian_traj(NUMBER_DAYS, S_0))

#%% Merging a simulation with the historical data

import pandas as pd

historical_dates = data1.index.tolist()
new_dates = pd.date_range(start=historical_dates[-1], periods=NUMBER_DAYS, freq='B')

new_data = pd.DataFrame(index=new_dates, columns=["Adj Close"])
new_data["Adj Close"] = brownian_traj(NUMBER_DAYS, S_0)

merged_data = pd.concat([data1, new_data])

plt.plot(data1.index, data1["Adj Close"], label="Historical Data", color="darkblue")
plt.plot(new_data.index, new_data["Adj Close"], label="Simulated Data", color="red")

plt.legend()
plt.show()

#%% Simulating multiple paths

def mult_brownian_paths(number_days, S_0, m):  #number_years = 1, 2 or 3... and trading_days = 252 generally
    paths = [] # List of the different prices paths
    lasts = [] # List of the last predicted price (for log-normality plot)
    delta_t = 1/(number_days)
    for i in range(m):
        simu = np.zeros(number_days)
        simu[0] = S_0
        S_new = S_0
        for k in range(1, number_days):
            eps = box_muller()
            S_new = S_new * np.exp(((mu - 0.5 * sigma**2) * delta_t) + sigma * eps * np.sqrt(delta_t))
            simu[k] = S_new
        paths.append(simu)
        lasts.append(simu[-1])
    return paths, lasts


S_0 = data1["Adj Close"][-1]
NUMBER_PATHS = 10**4

paths, lasts = mult_brownian_paths(NUMBER_DAYS, S_0, NUMBER_PATHS)
plt.figure()
for k in range(NUMBER_PATHS):
    plt.plot(paths[k])
plt.show()

# Prices following a log-normal distribution
plt.figure()
plt.hist(lasts, bins = 10**2)
plt.show()

# %%
