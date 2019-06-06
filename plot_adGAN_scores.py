import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import time
# from sklearn.metrics import roc_curve, auc
health = []
lesion = []
with open('/Users/chenjingkun/Documents/code/tools/health.txt', 'r') as f:
    for line in f:
        health.append(float(line))
with open('/Users/chenjingkun/Documents/code/tools/lesion.txt', 'r') as f:
    for line in f:
        lesion.append(float(line))

n, bins, patches = plt.hist(health, 50, facecolor='green',alpha=0.5)
n, bins, patches = plt.hist(lesion, 50, facecolor='red',alpha=0.5)

# add a 'best fit' line
# y = mlab.normpdf( bins)
# l = plt.plot(bins, y, 'r-', linewidth=1)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title(r'$\mathrm{Anogan-Mean}$')
# plt.axis('off')
plt.grid(True)

plt.show()

