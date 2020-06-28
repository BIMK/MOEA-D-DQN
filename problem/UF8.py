# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class UF8(ea.Problem): # 继承Problem父类
    def __init__(self):
        name      = 'UF8' # 初始化name（函数名称，可以随意设置）
        M         = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim       = 30  # 初始化Dim（决策变量维数）
        varTypes  = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb        = [0,0] + [-2] * (Dim - 2)  # 决策变量下界
        ub        = [1,1] + [2]  * (Dim - 2)  # 决策变量上界
        lbin      = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin      = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 决策变量矩阵
        x1   = Vars[:, [0]]
        x2   = Vars[:, [1]]
        J1   = np.array(list(range(3, self.Dim, 3)))
        J2   = np.array(list(range(4, self.Dim, 3)))
        J3   = np.array(list(range(2, self.Dim, 3)))
        # print(J1, J2, J3)
        J    = np.arange(1, 31)
        J    = J[np.newaxis, :]
        # f    = 2*np.mean((Vars-2*x2*np.sin(2*np.pi*x1+J*np.pi/self.Dim))**2 ,1,keepdims = True)
        f    = (Vars-2*x2*np.sin(2*np.pi*x1+J*np.pi/self.Dim))**2
        # print(f.shape)
        f1   = np.cos(0.5*x1*np.pi)*np.cos(0.5*x2*np.pi) + 2*np.mean(f[:, J1], 1, keepdims=True)
        f2   = np.cos(0.5*x1*np.pi)*np.sin(0.5*x2*np.pi) + 2*np.mean(f[:, J2], 1, keepdims=True)
        f3   = np.sin(0.5*x1*np.pi)                      + 2*np.mean(f[:, J3], 1, keepdims=True)
        pop.ObjV = np.hstack([f1, f2, f3])

    def calReferObjV(self):  # 理论最优值
        N = 10000 # 生成10000个参考点
        ObjV,N = ea.crtup(self.M, N)  # ObjV.shape=N,3
        ObjV = ObjV/np.sqrt(np.sum(ObjV**2,1,keepdims=True))
        referenceObjV = ObjV
        return referenceObjV

class pop():
    def __init__(self):
        self.Phen = None
        self.ObjV = None

if __name__ == '__main__':
    phen = scio.loadmat('phen.mat')
    u = UF8()
    p = pop()
    p.Phen = phen['phen']
    u.aimFunc(p)
    re = u.calReferObjV()
    # print(re.shape)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(re[:,0],re[:,1],re[:,2])
    plt.show()
    print(p.ObjV)

   








