# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

""" 
% No.   Name                              Samples Features Classes
% 1     Statlog_Australian                  690      14       2
% 2     Climate                             540      18       2
% 3     Statlog_German                     1000      24       2
% 4     Connectionist_bench_Sonar           208      60       2
"""


class NN(ea.Problem):
    def __init__(self, M=2, nHidden=20, dataNo=0) -> None:
        self.nHidden = nHidden
        name = 'NN'
        maxormins = [1] * M  # 最小化目标值
        datas_name = ['Statlog_Australian', 'Climate', 'Statlog_German', 'Connectionist_bench_Sonar']
        dataname = datas_name[dataNo]
        datasetNN = np.load('./data_set/DatasetNN.npy', allow_pickle=True)
        data = datasetNN[dataname][0][0]  # shape=690*15  最后一维是分类
        input_data = data[:, :-1]
        # print(np.shape(data))
        train_size = int(np.shape(data)[0] * 0.8)
        self.features = np.shape(data)[1] - 1
        mean = np.mean(input_data, axis=1, keepdims=True)
        std = np.std(input_data, axis=1, keepdims=True)
        input_data = (input_data - mean) / std
        output_data = data[:, -1:]
        self.trainIn = input_data[:train_size, :]
        self.trainOut = output_data[:train_size, :]
        self.testIn = input_data[train_size:, :]
        self.testOut = output_data[train_size:, :]
        Dim = (self.features + 1) * nHidden + (nHidden + 1) * 1   # 决策变量维度
        varTypes = [0] * Dim   # 0 实数， 1 整数
        lb = [-1] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def predict(self, x, w1, w2):
        n = np.shape(x)[0]
        Y = 1 - 2 / (1 + np.exp(2 * np.matmul(np.concatenate([np.ones((n, 1)), x], 1), w1)))
        Z = 1 / (1 + np.exp(np.matmul(-np.concatenate([np.ones((np.shape(Y)[0], 1)), Y], 1), w2)))
        return Z, Y

    def aimFunc(self, pop):  # 计算目标函数
        Vars = pop.Phen
        n = np.shape(Vars)[0]
        ObjV = np.zeros((n, 2))  # 目标函数，为什么是两目标
        for i in range(n):
            w1 = np.reshape(Vars[i, : (self.features + 1) * self.nHidden], (self.features + 1, self.nHidden))
            w2 = np.reshape(Vars[i, (self.features + 1) * self.nHidden:], (self.nHidden + 1, 1))
            z, y = self.predict(self.trainIn, w1, w2)
            ObjV[i, 0] = np.mean(np.abs(Vars[i, :]))
            z = np.round(z)  # 四舍五入
            ObjV[i, 1] = np.mean(z != self.trainOut)
        pop.ObjV = ObjV

    def calReferObjV(self):  # 计算参考值
        N = 1000  # 欲生成10000个全局帕累托最优解
        # 参数a,b,c为求解方程得到，详见DTLZ7的参考文献
        a = 0.2514118360889171
        b = 0.6316265307000614
        c = 0.8594008566447239
        Vars, Sizes = ea.crtgp(self.M - 1, N)  # 生成单位超空间内均匀的网格点集
        middle = 0.5
        left = Vars <= middle
        right = Vars > middle
        maxs_Left = np.max(Vars[left])
        if maxs_Left > 0:
            Vars[left] = Vars[left] / maxs_Left * a
        Vars[right] = (Vars[right] - middle) / (np.max(Vars[right]) - middle) * (c - b) + b
        P = np.hstack([Vars, (2 * self.M - np.sum(Vars * (1 + np.sin(3 * np.pi * Vars)), 1, keepdims=True))])
        referenceObjV = P
        return referenceObjV
