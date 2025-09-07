#%% European Options

import numpy as np
import matplotlib.pyplot as plt

K = 100
x = np.linspace(50, 150, 10**3)
ZEROS = np.zeros(10**3)

# Calculate the payoff for a long call option
y_long_call = np.maximum(x-K, ZEROS)
y_short_call = -np.maximum(x-K, ZEROS)
y_long_put = np.maximum(K-x, ZEROS)
y_short_put = -np.maximum(K-x, ZEROS)

#Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, y_long_call, color = "darkblue")
axs[0, 0].set_title('Long Call')
axs[0, 0].grid(True)
axs[0, 0].set_facecolor('lightgrey')

axs[0, 1].plot(x, y_short_call, color = "darkblue")
axs[0, 1].set_title('Short Call')
axs[0, 1].grid(True)
axs[0, 1].set_facecolor('lightgrey')

axs[1, 0].plot(x, y_long_put, color = "darkblue")
axs[1, 0].set_title('Long Put')
axs[1, 0].grid(True)
axs[1, 0].set_facecolor('lightgrey')

axs[1, 1].plot(x, y_short_put, color = "darkblue")
axs[1, 1].set_title('Short Put')
axs[1, 1].grid(True)
axs[1, 1].set_facecolor('lightgrey')

for ax in axs.flat:
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Payoff')

plt.tight_layout()
plt.savefig("Payoff_European")
plt.show()

# %%
