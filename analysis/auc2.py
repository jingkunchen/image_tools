# -*- coding: utf-8 -*-
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import heapq
ytrue = []
yscore = []
with open('patch_brain_lesion.txt', 'r') as f:
        for line in f:
            ytrue.append(0)
            yscore.append(float(line))

print(heapq.nlargest(2500, yscore))
print(heapq.nsmallest(2500, yscore))
# with open('patch_skin_lesion.txt', 'r') as f:
#         for line in f:
#             ytrue.append(1)
#             yscore.append(float(line))
# for i in range(200):
#     yscore.remove(max(yscore))
#     ytrue = ytrue[:len(ytrue)-1]
ytrue_tmp = []
yscore_tmp = []
with open('alocc_brain_lesion.txt', 'r') as f:
        for line in f:
            ytrue_tmp.append(1)
            yscore_tmp.append(float(line))
list_over = []
# print(heapq.nlargest(2682, yscore_tmp))
# print(heapq.nsmallest(2682, yscore_tmp))
print(len(yscore_tmp))
print(len(yscore))
count = 0
for i in range(0,5791):
    if yscore[i] > 0.6330365:
        # print(yscore[i] - yscore_tmp[i])
        count = count + 1
        if  yscore[i] - yscore_tmp[i]  > 0.02 :
            # print(yscore[i] - yscore_tmp[i] )
            list_over.append(i)
print(count)
print(list_over)
print(len(list_over))
# for i in range(200):
#     ytrue_tmp.remove(min(ytrue_tmp))
#     yscore_tmp = yscore_tmp[:len(yscore_tmp)-1]
# ytrue = ytrue + ytrue_tmp
# yscore = yscore + yscore_tmp
# print(max(yscore))
# print(min(yscore))

# fpr,tpr,threshold = roc_curve(ytrue, yscore, pos_label=1)
# # print("fpr:",fpr)
# # print("tpr:",tpr)
# # print("threshold:",threshold)

# roc_auc = auc(fpr,tpr)

# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#                  lw=lw, label='ROC curve (area = %f)' % roc_auc)

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('brain patch')
# plt.legend(loc="lower right")


# plt.show()
