import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import time
import heapq
# from sklearn.metrics import roc_curve, auc
health = []
lesion = []
#skin
# count = 4285
# count = 3000
#brain
count = 2200
count_health_alocc = 0
count_lesion_alocc = 0
with open('autoencoder_brain_health.txt', 'r') as f:
    for line in f:
        if count_health_alocc == count:
            break
        else:
            count_health_alocc = count_health_alocc + 1
            # tmp = heapq.nlargest(4, (float(line.split()[0]),float(line.split()[1]),float(line.split()[2]),float(line.split()[3])))
            # health.append(tmp[0]+tmp[1]+tmp[2]++tmp[3])
            health.append(float(line.split()[0]))
with open('autoencoder_brain_lesion.txt', 'r') as f:

    for line in f:
        if count_lesion_alocc == count:
            break
        else:
            count_lesion_alocc = count_lesion_alocc + 1
            # tmp = heapq.nlargest(4, (float(line.split()[0]),float(line.split()[1]),float(line.split()[2]),float(line.split()[3])))
            # lesion.append(tmp[0]+tmp[1]+tmp[2]++tmp[3])
            lesion.append(float(line.split()[0]))





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
plt.title(r'$\mathrm{autoencoder-BraTS}$')
# plt.axis('off')
plt.grid(True)

plt.show()

