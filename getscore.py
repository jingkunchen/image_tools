import os
import re
sample = '499score:'




#list=[i.start() for i in re.finditer('\\\\', 'C:\\Users\\aaa\\computer\\flicker\\01213.jpg')]
#print(list)

score1 = []
score2 = []
score3 = []
score4 = []
score5 = []
'''
with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_lesion_test_1.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            #print count
            print float(line.replace('499score:',''))
'''
with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_health_test_1.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            score1.append(line.replace('499score:',''))

with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_health_test_2.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            score2.append(line.replace('499score:',''))
with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_health_test_3.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            score3.append(line.replace('499score:',''))
with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_health_test_4.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            score4.append(line.replace('499score:',''))
with open('/Users/chenjingkun/Documents/result/anogan/brain/brain_health_test_5.log', 'r') as f:
    count = 0
    for line in f:
        if line.find(sample) == 0:
            count = count + 1
            #print count
            score5.append(line.replace('499score:',''))


count = 0            
for i in range(28878):
    count = count + 1
    #print count
    
    print (float(score1[i])+float(score2[i])+float(score3[i])+float(score4[i])+float(score5[i]))/5
            #!print count
            #list= [i.start() for i in re.finditer('\,', score)]
            #print float(score[0:list[0]]
            #print type(score)

#Epoch: [99], anomaly score:
