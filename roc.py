# -*- coding: utf-8 -*-
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import auc

ytrue = []
yscore = []
with open('/Users/chenjingkun/Documents/result/newkde/health.txt', 'r') as f:
        for line in f:
            ytrue.append(0)
            yscore.append(1-float(line))

with open('/Users/chenjingkun/Documents/result/newkde/lesion.txt', 'r') as f:
        for line in f:
            ytrue.append(1)
            yscore.append(1-float(line))

print ytrue
print yscore

fpr,tpr,threshold = roc_curve(ytrue, yscore, pos_label=1)
print "fpr:",fpr
print "tpr:",tpr
print "threshold:",threshold

roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('skin anomoly detection')
plt.legend(loc="lower right")


plt.show()
