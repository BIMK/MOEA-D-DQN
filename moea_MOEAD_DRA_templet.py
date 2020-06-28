# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # 导入geatpy库
from scipy.spatial.distance import cdist
from sys import path as paths
from os import path
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])
from matplotlib import pyplot as plt
from mut_de import Best_mut,DE_rand_1,DE_rand_2,DE_current_to_rand_1,DE_current_to_rand_2,RL_mut_moea
import random
import sys

class moea_MOEAD_DRA_templet(ea.MoeaAlgorithm):
    
    """
moea_MOEAD_DRA_templet : class - 带动态资源分配的多目标进化MOEA/D-DRA算法模板,不处理约束
    
算法描述:
    和MOEA/D最大的不同就是每次进化不是所有的个体，而是选出[N/5]-m个体构成I，定义一个pi指标
    表示子问题的优化程度，优化程度高的个体将被优先加入I。

参考文献:
    [1] The Performance of a New Version of MOEA/D on CEC09 Unconstrained MOP Test Instances

    """
    
    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population) # 先调用父类构造方法
        self.name = 'MOEA/D-DRA'
        self.uniformPoint, self.NIND = ea.crtup(self.problem.M, population.sizes) # 生成在单位目标维度上均匀分布的参考点集
        self.mutDE = [DE_rand_1(), DE_rand_2(), DE_current_to_rand_1(), DE_current_to_rand_2()]  # 差分算子
        # self.mutDE = RL_mut_moea(problem, self.uniformPoint)
        # self.mutDE = Best_mut(problem, self.uniformPoint)
        # self.recOper = ea.Recsbx(XOVR=1,n=20,Half=True) # 模拟二进制交叉
        self.mutPolyn = ea.Mutpolyn(Pm = 1/self.problem.Dim, DisI = 20, FixType=4) # 生成多项式变异算子对象
        self.neighborSize = self.NIND//10 # 邻域大小，当设置为None时，将会自动设置为等于种群规模
        self.neighborSize = max(self.neighborSize, 2) # 确保不小于2
        self.nr = self.NIND//100  #max number of solutions replaced by each offspring
        self.nr = max(self.nr, 2) # 确保nr不小于2
        self.decomposition = ea.tcheby # 采用切比雪夫权重聚合法
        # self.decomposition = ea.pbi # 采用pbi权重聚合法
        self.delta = 0.9 # (Probability of Selection)表示进化时有多大的概率只从邻域中选择个体参与进化
    
    def reinsertion(self, indices, population, offspring, idealPoint, referPoint):
        
        """
        描述:
            重插入更新种群个体。
            indices: 邻居索引
            idealPoint: 每个目标上的理想点值
            referPoint: 均匀分布的参考点集
        """
        
        weights = referPoint[indices, :]
        pop_ObjV = population.ObjV[indices, :] # 获取邻居个体的目标函数值
        pop_CV = population.CV[indices, :] if population.CV is not None else None # 获取邻居个体的违反约束程度矩阵
        # 计算切比雪夫距离
        CombinObjV = self.decomposition(pop_ObjV, weights, idealPoint, pop_CV, self.problem.maxormins)
        off_CombinObjV = self.decomposition(offspring.ObjV, weights, idealPoint, offspring.CV, self.problem.maxormins)
        # 更新至多nr个个体
        population[indices[np.where(off_CombinObjV <= CombinObjV)[0][:self.nr]]] = offspring
        # 更新全部个体
        # population[indices[np.where(off_CombinObjV <= CombinObjV)[0]]] = offspring
    
    def tournamentSelection(self, K, N, pi):
        """
        竞赛选择
        K元竞赛选择，每次抽取K个个体，选出其中目标值最大的。
        执行N次，返回的索引长度应当是N
        return: list
        """
        ind = []
        for _ in range(N):
            # 采样K个
            parent = np.random.choice(self.NIND, K, replace=False)  # replace=False不重复采样
            # 找出K个中pi最大的元素在种群中的索引, pi值越大，表示在这个子问题上的提升越明显
            maxind = parent[np.argmax(pi[parent])]
            ind.append(maxind)
        return ind

    
    def run(self, prophetPop = None): # prophetPop为先知种群（即包含先验知识的种群）
        #==========================初始化配置===========================
        population = self.population
        uniformPoint = self.uniformPoint
        NIND = self.NIND
        self.initialization() # 初始化算法模板的一些动态参数
        #===========================准备进化============================
        # uniformPoint, NIND = ea.crtup(self.problem.M, population.sizes) # 生成在单位目标维度上均匀分布的参考点集
        population.initChrom(self.NIND)   # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        self.call_aimFunc(population) # 计算种群的目标函数值
        # 生成由所有邻居索引组成的矩阵
        neighborIdx = np.argsort(cdist(uniformPoint, uniformPoint), axis=1, kind='mergesort')[:, :self.neighborSize]
        # 计算理想点
        idealPoint = ea.crtidp(population.ObjV, population.CV, self.problem.maxormins)
        pi = np.ones((NIND,1))
        # old Techbycheff function value of each solution on its subproblem 当前每个子问题的目标值
        oldz = ea.tcheby(population.ObjV,uniformPoint,idealPoint)
        # PopCountOpers = []  #统计不同进化阶段算子选择的结果
        # CountOpers = np.zeros((self.NIND,self.mutDE.n))  # 统计不同子问题算子选择结果

        #===========================开始进化============================
        while self.terminated(population) == False:
            for _ in range(5): # 子种群大小是[N/5]-m
                # 这里的边界是指weight在M-1个方向上权重为0
                Bounday = np.where(np.sum(self.uniformPoint<0.0001,1)==(self.problem.M-1))[0].tolist()
                # 利用10元竞赛选出pi最大的N/5个体的索引
                I = self.tournamentSelection(10, NIND//5-self.problem.M,pi)+Bounday
                select_rands = np.random.rand(NIND)
                for i in I:
                    if select_rands[i] < self.delta:
                        indices = neighborIdx[i, :]  # 得到邻居索引
                    else:
                        indices = np.arange(NIND)
                    np.random.shuffle(indices)
                    offspring = ea.Population(population.Encoding, population.Field, 1) # 实例化一个种群对象用于存储进化的后代（这里只进化生成一个后代）
                    # 对选出的个体进行进化操作
                    # offspring.Chrom = self.mutDE.do(population.Encoding, population.Chrom, population.Field, i, indices,idealPoint)
                    offspring.Chrom = self.mutDE[0].do(population.Chrom, population.Field, i, indices)
                    offspring.Chrom = self.mutPolyn.do(offspring.Encoding, offspring.Chrom, offspring.Field) # 变异
                    self.call_aimFunc(offspring) # 求进化后个体的目标函数值
                    # 更新理想点
                    idealPoint = ea.crtidp(offspring.ObjV, offspring.CV, self.problem.maxormins, idealPoint)
                    # 重插入更新种群个体
                    self.reinsertion(indices, population, offspring, idealPoint, uniformPoint)
                    # 统计不同子问题的算子选择结果
                    # CountOpers[i] += self.mutDE.CountOpers
                    # self.mutDE.CountOpers = np.zeros(self.mutDE.n)
            """每10代更新一次pi值"""
            if self.currentGen % 10 == 0:
                newz = ea.tcheby(population.ObjV, uniformPoint, idealPoint)
                DELTA = (oldz-newz)/oldz
                idx = DELTA < 0.001
                pi[~idx] = 1
                pi[idx] = (0.95+0.05*DELTA[idx]/0.001)*pi[idx]
                oldz = newz
                """统计不同进化阶段算子选择的结果"""
                # PopCountOpers.append(self.mutDE.CountOpers)
                # self.mutDE.CountOpers = np.zeros(self.mutDE.n) # 清空算子选择记录器
        
        # 画出不同进化阶段算子选择的结果
        # BestSelection = np.array(PopCountOpers)
        # 画出不同子问题算子选择的结果
        # BestSelection = CountOpers
        # for i in range(self.mutDE.n):
        #     plt.plot(BestSelection[:,i],'.',label=self.mutDE.mutOper[i].name)
        # plt.legend()
        # plt.show()

        return self.finishing(population), population # 调用finishing完成后续工作并返回结果
    