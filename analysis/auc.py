# -*- coding: utf-8 -*-
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import heapq
import scipy.stats as stats
ytrue = []
yscore = []
ytrue2 = []
yscore2 = []
count_health = 0
count_lesion = 0
dataset_name ='skin'
#skin
# count = 4285
count = 3000
#brain
# count = 2200


with open('patch_'+dataset_name+'_health_g_patch.txt', 'r') as f:
        for line in f:
            if count_health == count:
                break
            else:
                count_health = count_health + 1
                ytrue.append(0)
                ytrue2.append(0)
                yscore2.append((float(line.split()[0])+float(line.split()[1])+float(line.split()[2])+float(line.split()[3]))/4)
                #patch
                tmp = heapq.nlargest(4, (float(line.split()[0]),float(line.split()[1]),float(line.split()[2]),float(line.split()[3])))
                yscore.append(1.1*tmp[0]+tmp[1])
with open('patch_'+dataset_name+'_lesion_g_patch.txt', 'r') as f:
        for line in f:
            if count_lesion == count:
                break
            else:
                count_lesion = count_lesion + 1
                ytrue.append(1)
                ytrue2.append(1)
                yscore2.append((float(line.split()[0])+float(line.split()[1])+float(line.split()[2])+float(line.split()[3]))/4)
                #patch
                tmp = heapq.nlargest(4, (float(line.split()[0]),float(line.split()[1]),float(line.split()[2]),float(line.split()[3])))
                yscore.append(1.1*tmp[0]+tmp[1])



print(count_health,count_lesion)
fpr,tpr,threshold = roc_curve(ytrue, yscore, pos_label=1)
print("fpr:",fpr)
print("tpr:",tpr)
print("threshold:",threshold)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='patch with ranking (area = %.4f)' % roc_auc)


fpr2,tpr2,threshold2 = roc_curve(ytrue2, yscore2, pos_label=1)
roc_auc2 = auc(fpr2,tpr2)



plt.plot(fpr2, tpr2, color='green',
                 lw=lw, label='patch without ranking (area = %.4f)' % roc_auc2)

ytrue1 = []
yscore1 = []
count_health_alocc = 0
count_lesion_alocc = 0


with open('alocc_'+dataset_name+'_health.txt', 'r') as f:
        for line in f:
            if count_health_alocc == count:
                break
            else:
                count_health_alocc = count_health_alocc + 1
                ytrue1.append(0)
                yscore1.append(float(line.split()[0]))
with open('alocc_'+dataset_name+'_lesion.txt', 'r') as f:
        for line in f:
            if count_lesion_alocc == count:
                break
            else:
                count_lesion_alocc = count_lesion_alocc + 1
                ytrue1.append(1)
                yscore1.append(float(line.split()[0]))
print(count_health_alocc, count_lesion_alocc)

fpr1,tpr1,threshold1 = roc_curve(ytrue1, yscore1, pos_label=1)

fpr_tmp = [round(i,4) for i in fpr]
fpr_tmp1 = [round(i,4) for i in fpr1]
print(fpr_tmp,fpr_tmp1)
a = [x for x in fpr_tmp if x in fpr_tmp1]
print(a)
list1 = []
list2 = []
for i in range(0,70):
    list1.append(tpr1[fpr_tmp.index(a[i])])
    list2.append(tpr1[fpr_tmp1.index(a[i])])
print(list1)
print(list2)
print(stats.wilcoxon(list1, list2))
print(stats.ranksums(list1, list2))
roc_auc1 = auc(fpr1,tpr1)
lw = 2
plt.plot(fpr1, tpr1, color='blue',
                 lw=lw, label='alocc (area = %.4f)' % roc_auc1)

ytrue3 = []
yscore3 = []
count_health_svm = 0
count_lesion_svm = 0
with open('svm_'+dataset_name+'_health.txt', 'r') as f:
        for line in f:
            if count_health_svm == count:
                break
            else:
                count_health_svm = count_health_svm + 1
                ytrue3.append(1)
                print(line)
                yscore3.append(float(line.split()[1]))
with open('svm_'+dataset_name+'_lesion.txt', 'r') as f:
        for line in f:
            if count_lesion_svm == count:
                break
            else:
                count_lesion_svm = count_lesion_svm + 1
                ytrue3.append(0)
                yscore3.append(float(line.split()[1]))
print(count_health_svm, count_lesion_svm)

fpr3, tpr3, threshold3 = roc_curve(ytrue3, yscore3, pos_label=1)
roc_auc3 = auc(fpr3,tpr3)
lw = 2
plt.plot(fpr3, tpr3, color='red',
                 lw=lw, label='svm (area = %.4f)' % roc_auc3)


ytrue4 = []
yscore4 = []
count_health_autoencoder = 0
count_lesion_autoencoder = 0
with open('autoencoder_'+dataset_name+'_health.txt', 'r') as f:
        for line in f:
            if count_health_autoencoder == count:
                break
            else:
                count_health_autoencoder = count_health_autoencoder + 1
                ytrue4.append(0)
                print(line)
                yscore4.append(float(line.split()[0]))

with open('autoencoder_'+dataset_name+'_lesion.txt', 'r') as f:
        for line in f:
            if count_lesion_autoencoder == count:
                break
            else:
                count_lesion_autoencoder = count_lesion_autoencoder + 1
                ytrue4.append(1)
                yscore4.append(float(line.split()[0]))
print(count_health_autoencoder, count_lesion_autoencoder)

fpr4, tpr4, threshold4 = roc_curve(ytrue4, yscore4, pos_label=1)
roc_auc4 = auc(fpr4,tpr4)
lw = 2
plt.plot(fpr4, tpr4, color='black',
                 lw=lw, label='autoencoder (area = %.4f)' % roc_auc4)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ISIC')
plt.legend(loc="lower right")


plt.show()
