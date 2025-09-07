import matplotlib.pyplot as plt
import numpy as np

# Data
option_types = ['European', 'Asian', 'Barrier', 'American']
methods = ['Standard MC', 'Antithetic Variables', 'Control Variables', 'Combined']

pt_ratios = {
    'European': [0.0392, 0.0778, 0.5213, 0.5244],
    'Asian': [0.0299, 0.0782, 1.7967, 3.4033],
    'Barrier': [0.0394, 0.0538, 0, 0],
    'American': [0.3449, 0.6803, 0.1855, 0.3629]
}

# Creating the bar chart
x = np.arange(len(option_types))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, method in enumerate(methods):
    ax.bar(x + i*width, [pt_ratios[opt][i] for opt in option_types], width, label=method)

# Adding labels and title
ax.set_xlabel('Option Types')
ax.set_ylabel('P/T Ratio')
ax.set_title('Efficiency of Variance Reduction Methods Across Different Option Types')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(option_types)
ax.legend()

plt.show()
