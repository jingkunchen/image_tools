import numpy as np
import random
import math

random.seed(1)
np.random.seed(1)


"""  
    函数功能：选择初始中心点  
    points: 数据集 
    pNum: 数据集样本个数 
    cNum: 选取聚类中心个数
"""

def initCenters(points, pNum, cNum):
    #初始中心点列表
    centers = []
    #在样本中随机选取一个点作为第一个中心点
    firstCenterIndex = random.randint(0, pNum - 1)
    centers.append(points[firstCenterIndex])
    #初始距离列表
    distance = []
    #对于每个中心点类别
    for cIndex in range(1, cNum):
        print("cIndex:",cIndex)
        #sum为数据集中每个样本点和其最近的中心点的距离和
        sum = 0.0
        #遍历整个数据集
        for pIndex in range(0, pNum):
            print("pNum1:",pIndex)
            #计算每个样本和最近的中心点的距离
            dist = nearest(points[pIndex], centers, cIndex)
            #将距离存到距离列表中
            distance.append(dist)
            #距离相加
            sum += dist
        #随机在（0，sum）中间取一个数值
        ran = random.uniform(0, sum)
        #遍历数据集
        for pIndex in range(0, pNum):
            print("pNum2:",pIndex)
            #ran-=D(x)
            ran -= distance[pIndex]
            if ran > 0: continue
            centers.append(points[pIndex])
            break
    return centers


""" 
    函数功能：计算点和中心之间的最小距离 
    point: 数据点 
    centers: 已经选择的中心
    cIndex: 已经选择的中心个数 
"""


def nearest(point, centers, cIndex):
    #初始一个足够大的最小距离
    minDist = 65536.0
    dist = 0.0
    for index in range(0, cIndex):
        dist = distance(point, centers[index])
        if minDist > dist:
            minDist = dist
    return minDist



""" 
    函数功能：计算点和中心之间的距离 
    point: 数据点 
    center:中心 
     
"""
def distance(point, center):
    dim = len(point)
    if dim != len(center):
        return 0.0
    a = 0.0
    b = 0.0
    c = 0.0
    for index in range(0, dim):
        a += point[index] * center[index]
        b += math.pow(point[index], 2)
        c += math.pow(center[index], 2)
    b = math.sqrt(b)
    c = math.sqrt(c)
    try:
        return a / (b * c)
    except Exception as e:
        print(e)
    return 0.0

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    sse = 0
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
        sse += min_dist
    return idx, sse


def compute_centroids(X, idx, k):
    m, n = X.shape

    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) /
                           len(indices[0])).ravel()
    return centroids


def run_k_means(X, initial_centroids, max_iters):
    global sse
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)

    centroids = initial_centroids

    for i in range(max_iters):
        print("run_k_means:",i)
        idx, sse = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids, sse

X_train = np.load("/Users/chenjingkun/Documents/data/skin/skin_health_train.npy")
X_train = X_train / 255.

print("X_train:",X_train.shape)
cluster_data = X_train.reshape(-1, 32 * 32 *3)
cluster_show = X_train

m = cluster_data.shape[0]
print(cluster_data.shape)
cluster_points = []
for i in range(m):
    cluster_points.append(cluster_data[i, :])
cNum = 3
print("initial_centroids")
initial_centroids = np.array(initCenters(cluster_points, m, cNum))
max_iters = 3
print("run_k_means")
idx, centroids, sse = run_k_means(cluster_data, initial_centroids,
                                max_iters)
np.save("skin_centroids_3cluster.npy", centroids)
ones = np.ones((1, 32, 32, 1))
zeros = np.zeros((1, 32, 32, 1))
label_0 =  np.concatenate((np.concatenate((ones, zeros), axis=3), zeros), axis=3)

label_1 =  np.concatenate((np.concatenate((zeros, ones), axis=3), zeros), axis=3)

label_2 =  np.concatenate((np.concatenate((zeros, zeros), axis=3), ones), axis=3)

if (int(idx[0]) == 0):
    labels =  label_0
if (int(idx[0]) == 1):
    labels =  label_1
if (int(idx[0]) == 2):
    labels =  label_2
for i in range(1, X_train.shape[0]):
    if (int(idx[i]) == 0):
        labels = np.concatenate((labels, label_0), axis=0)
    if (int(idx[i]) == 1):
        labels = np.concatenate((labels, label_1), axis=0)
    if (int(idx[i]) == 2):
        labels = np.concatenate((labels, label_2), axis=0)
    print(labels.shape)
labeled_data = np.concatenate((X_train, labels), axis=3)
np.save("skin_labeled_data_3cluster.npy", labeled_data)