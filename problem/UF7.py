# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import scipy.io as scio
import matplotlib.pyplot as plt

class UF7(ea.Problem): # 继承Problem父类
    def __init__(self):
        name      = 'UF7' # 初始化name（函数名称，可以随意设置）
        M         = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim       = 30  # 初始化Dim（决策变量维数）
        varTypes  = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb        = [0] + [-1] * (Dim - 1)  # 决策变量下界
        ub        = [1] * Dim # 决策变量上界
        lbin      = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin      = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):  # 目标函数
        Vars  = pop.Phen  # 决策变量矩阵
        x1    = Vars[:, [0]]
        J1    = np.array(list(range(2, self.Dim, 2)))
        J2    = np.array(list(range(1, self.Dim, 2)))
        J     = np.arange(1, 31)
        J     = J[np.newaxis, :]
        y     = Vars - np.sin(6*np.pi*x1 + (J*np.pi)/self.Dim)
        yJ1   = y[:, J1]
        yJ2   = y[:, J2]
        f1 = x1**0.2   + 2*np.mean(yJ1**2,1,keepdims=True)
        f2 = 1-x1**0.2 + 2*np.mean(yJ2**2,1,keepdims=True)
        pop.ObjV = np.hstack([f1, f2])

    def calReferObjV(self):  # 理论最优值
        N = 10000 # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

class pop():
    def __init__(self):
        self.Phen = None
        self.ObjV = None

if __name__ == '__main__':
    phen = scio.loadmat('phen.mat')
    u = UF7()
    p = pop()
    p.Phen = phen['phen']
    u.aimFunc(p)
    re = u.calReferObjV()
    # print(re.shape)
    plt.scatter(re[:,0],re[:,1],s=0.1)
    plt.show()
    print(p.ObjV)

   








