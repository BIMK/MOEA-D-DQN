# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # 导入geatpy库
from scipy.spatial.distance import cdist
from sys import path as paths
from os import path
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])
from matplotlib import pyplot as plt
from mut_de import DE_rand_1, DE_rand_2, DE_current_to_rand_1, DE_current_to_rand_2, Best_mut, RL_mut_moea
import random
import sys

class moea_MOEAD_templet(ea.MoeaAlgorithm):
    
    """
moea_MOEAD_templet : class - 多目标进化MOEA/D算法模板,不处理约束
    
算法描述:
    采用MOEA/D（不设全局最优存档）进行多目标优化，算法详见参考文献[1]。
    注：由于MOEA/D在每一代中需要循环地每次从种群中选择2个个体执行进化操作，因此在Python上，MOEA/D的性能会大幅度降低。

参考文献:
    [1] Qingfu Zhang, Hui Li. MOEA/D: A Multiobjective Evolutionary Algorithm 
    Based on Decomposition[M]. IEEE Press, 2007.

    """
    
    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population) # 先调用父类构造方法
        self.name = 'MOEA/D'
        self.uniformPoint, self.NIND = ea.crtup(self.problem.M, population.sizes) # 生成在单位目标维度上均匀分布的参考点集
        self.mutDE = [DE_rand_1(), DE_rand_2(), DE_current_to_rand_1(), DE_current_to_rand_2()]  # 差分算子
        # self.mutDE = Best_mut(problem, self.uniformPoint)
        # self.mutDE = RL_mut_moea(problem, self.uniformPoint)
        self.mutPolyn = ea.Mutpolyn(Pm = 1/self.problem.Dim, DisI = 20, FixType=4) # 生成多项式变异算子对象
        self.neighborSize = self.NIND//10 # 邻域大小，当设置为None时，将会自动设置为等于种群规模
        self.neighborSize = max(self.neighborSize, 2) # 确保不小于2
        self.nr = self.NIND//100  #max number of solutions replaced by each offspring
        self.nr = max(self.nr, 2) # 确保nr不小于2
        self.decomposition = ea.tcheby # 采用切比雪夫权重聚合法
        # self.decomposition = ea.pbi # 采用pbi权重聚合法
        self.delta = 0.9 # (Probability of Selection)表示进化时有多大的概率只从邻域中选择个体参与进化
    
    def reinsertion(self, indices, population, offspring, idealPoint):
        
        """
        描述:
            重插入更新种群个体。
            indices: 邻居索引
            idealPoint: 每个目标上的理想点值
        """
        
        weights = self.uniformPoint[indices, :]  # 和邻居比较Tech，或许可以替换邻居，但最多替换nr个
        pop_ObjV = population.ObjV[indices, :] # 获取邻居个体的目标函数值
        pop_CV = population.CV[indices, :] if population.CV is not None else None # 获取邻居个体的违反约束程度矩阵
        # 计算切比雪夫距离
        CombinObjV = self.decomposition(pop_ObjV, weights, idealPoint, pop_CV, self.problem.maxormins)
        off_CombinObjV = self.decomposition(offspring.ObjV, weights, idealPoint, offspring.CV, self.problem.maxormins)
        # 更新所有个体，而不是只更新n_r个
        # population[indices[np.where(off_CombinObjV <= CombinObjV)[0]]] = offspring
        # 更新至多nr个个体
        population[indices[np.where(off_CombinObjV <= CombinObjV)[0][:self.nr]]] = offspring
    
    def run(self, prophetPop = None): # prophetPop为先知种群（即包含先验知识的种群）
        #==========================初始化配置===========================
        population = self.population
        self.initialization() # 初始化算法模板的一些动态参数
        #===========================准备进化============================
        population.initChrom(self.NIND)   # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        self.call_aimFunc(population) # 计算种群的目标函数值
        # 生成由所有邻居索引组成的矩阵
        neighborIdx = np.argsort(cdist(self.uniformPoint, self.uniformPoint), axis=1, kind='mergesort')[:, :self.neighborSize]
        # 计算理想点
        idealPoint = ea.crtidp(population.ObjV, population.CV, self.problem.maxormins)
        # PopCountOpers = []  #统计不同进化阶段算子选择的结果
        # CountOpers = np.zeros((self.NIND,self.mutDE.n))  # 统计不同子问题算子选择结果
        #===========================开始进化============================
        while self.terminated(population) == False:
            select_rands = np.random.rand(self.NIND)  # 生成一组随机数
            for i in range(population.sizes):
                if select_rands[i] < self.delta:  # 从邻域中选择父代
                    indices = neighborIdx[i, :]   # 得到邻居索引
                else:                             # 父代可以从整个种群选择
                    indices = np.arange(self.NIND)                             
                np.random.shuffle(indices)        # 打乱交配池索引顺序
                offspring = ea.Population(population.Encoding, population.Field, 1) # 实例化一个种群对象用于存储进化的后代（这里只进化生成一个后代）
                # 对选出的个体进行进化操作
                # offspring.Chrom = self.mutDE.do(population.Encoding, population.Chrom, population.Field, i, indices,idealPoint)
                offspring.Chrom = self.mutDE[0].do(population.Chrom, population.Field, i, indices)
                offspring.Chrom = self.mutPolyn.do(offspring.Encoding, offspring.Chrom, offspring.Field) # 变异
                self.call_aimFunc(offspring) # 求进化后个体的目标函数值
                # 更新理想点
                idealPoint = ea.crtidp(offspring.ObjV, offspring.CV, self.problem.maxormins, idealPoint)
                # 重插入更新种群个体
                self.reinsertion(indices, population, offspring, idealPoint)
                # 画出同一个体在不同进化阶段的选择结果
                # CountOpers[i] += self.mutDE.CountOpers
                # self.mutDE.CountOpers = np.zeros(self.mutDE.n)
            # 每10代记录一次算子选择的结果
            # if self.currentGen % 1 == 0:
            #     PopCountOpers.append(self.mutDE.CountOpers)
            #     self.mutDE.CountOpers = np.zeros(self.mutDE.n)

        # 画出不同进化阶段算子选择的结果
        # BestSelection = np.array(PopCountOpers)
        # 画出不同参考方向算子选择的结果
        # BestSelection = CountOpers
        # for i in range(self.mutDE.n):
        #     plt.plot(BestSelection[:,i],'.',label=self.mutDE.mutOper[i].name)
        # plt.legend()
        # plt.show()

        # ea.moeaplot(population.ObjV, 'rand_1', saveFlag=False, gridFlag=True)
        return self.finishing(population), population # 调用finishing完成后续工作并返回结果
    