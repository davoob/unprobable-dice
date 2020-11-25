import numpy as np
import matplotlib.pyplot as plt

vals_probs = np.loadtxt('distribution.txt')
print(vals_probs)

probs = vals_probs[:, 1]
vals = vals_probs[:, 0]

mean = 0
for i, val in enumerate(vals):
    prob = probs[i]
    mean += prob*val
print(mean)

# plt.plot(vals, probs)
# plt.show()

deviations = []
for min_dice in range(10):
    discrete_probs_float = (min_dice+1) * probs/probs[0]
    discrete_probs = np.round(discrete_probs_float).astype(int)
    cur_deviations = discrete_probs_float - discrete_probs
    cur_deviation = np.sum(np.square(cur_deviations))
    deviations.append(cur_deviation)

print(deviations)

min_dice = np.argmin(deviations)
min_deviation = deviations[min_dice]

discrete_probs = np.round((min_dice+1) * probs/probs[0]).astype(int)
print(min_dice+1, discrete_probs, deviations[min_dice])