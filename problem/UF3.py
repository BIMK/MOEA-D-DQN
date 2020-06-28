# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import scipy.io as scio
import matplotlib.pyplot as plt

class UF3(ea.Problem):  # 继承Problem的父类
    def __init__(self):
        name = 'UF3'
        M = 2  # 目标维数
        maxormins = [1]*M   # 目标最大化标记
        Dim = 30   # 决策变量维数
        varTypes = [0] * Dim  # 变量类型实数
        lb = [0]*Dim  # 决策变量下界
        ub = [1]*Dim  # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.Dim ,2)))
        J2 = np.array(list(range(1, self.Dim ,2)))
        J = np.arange(1,31)
        J = J[np.newaxis, :]
        y = Vars - x1**(0.5*(1+(3*(J-2)/(self.Dim-2))))
        # print("y====")
        # print(y)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1            + (2/len(J1))*(4*np.sum(yJ1**2,1,keepdims=True) - \
            2*(np.prod(np.cos((20*yJ1*np.pi)/(np.sqrt(J1))),1,keepdims=True))+2) 
        f2 = 1-np.sqrt(x1) + (2/len(J2))*(4*np.sum(yJ2**2,1,keepdims=True) - \
            2*(np.prod(np.cos((20*yJ2*np.pi)/(np.sqrt(J2))),1,keepdims=True))+2) 
        pop.ObjV = np.hstack([f1, f2])

    def calReferObjV(self):  # 理论最优值
        N = 10000 # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV



class pop():
    def __init__(self):
        self.Phen = None
        self.ObjV = None

if __name__ == '__main__':
    phen = scio.loadmat('phen.mat')

    u = UF3()
    p = pop()
    p.Phen = phen['phen']
    u.aimFunc(p)
    re = u.calReferObjV()
    print(p.ObjV)
    plt.scatter(re[:,0],re[:,1],s=1)
    plt.show()
