# -*- coding: utf-8 -*-
import numpy as np

""" 生成2维权重向量 """
def initWeightVector(N):
    # +2是为了避免在某一目标上为0的情况
    N += 2
    lambda_ = np.zeros((N, 2))
    # 生成2目标权重向量
    for i in range(N):
        t = i / (N - 1)
        lambda_[i][0] = t
        lambda_[i][1] = 1 - t
    return lambda_[0:N-2]


"""计算两个向量间距离"""
def disVector(v1, v2):
    dim = len(v1)
    sum_ = 0
    for i in range(dim):
        sum_ += (v1[i] - v2[i]) ** 2
    return sum_


""" 生成邻域关系 """
def initNeighbourVector(lambda_, T):
    N = len(lambda_)
    dis = np.zeros([N, N])
    # 根据dis，选择距离每个向量最近的T个向量
    neighborVector_ = np.zeros([N, T], dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            dis[i][j] = dis[j][i] = disVector(lambda_[i], lambda_[j])
        neighborVector_[i] = dis[i].argsort()[0:T]
    return neighborVector_


"""利用Tchebycheff分解法计算目标函数值
return the max lambda_i*abs(ObjV_i-z_i)
"""
def Tche(lambda_, ObjV, z):
    diff = abs(ObjV - z)
    ans = lambda_ * diff
    return np.max(ans,  keepdims=True)


if __name__ == '__main__':
    import geatpy as ea
    weight = np.array([[0.7,0.3],[0.8,0.2]])
    ObjV = np.array([[1.1,2.7],[24,43]])
    z = np.array([1,1])
    t1 = Tche(weight, ObjV, z)
    print(t1)
    t2 = ea.tcheby(ObjV, weight, z)
    print(t2)

