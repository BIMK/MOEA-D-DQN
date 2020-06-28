# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # 导入geatpy库
from scipy.spatial.distance import cdist
from sys import path as paths
from os import path
from matplotlib import pyplot as plt
from mut_de import DE_rand_1, DE_rand_2, DE_current_to_rand_1, DE_current_to_rand_2, RL_mut_moea, Best_mut
from Nature_DQN import DQN
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])

class moea_MOEAD_DE_templet(ea.MoeaAlgorithm):
    
    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population) # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'MOEA/D-DE'
        self.uniformPoint, self.NIND = ea.crtup(self.problem.M, population.sizes) # 生成在单位目标维度上均匀分布的参考点集
        self.DQN = DQN(problem.Dim, 4)
        if population.Encoding == 'RI':
            # self.mutDE = [DE_rand_1(),DE_rand_2(),DE_current_to_rand_1(),DE_current_to_rand_2()]
            self.mutDE = RL_mut_moea(problem, self.uniformPoint, self.DQN)
            # self.mutDE = Best_mut(problem, self.uniformPoint)
            self.mutPolyn = ea.Mutpolyn(Pm = 1/self.problem.Dim, DisI = 20, FixType=4) # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''RI''.')
        if self.problem.M <= 2:
            self.decomposition = ea.tcheby # 采用切比雪夫权重聚合法
        else:
            self.decomposition = ea.pbi # 采用pbi权重聚合法
        self.Ps = 0.9 # (Probability of Selection)表示进化时有多大的概率只从邻域中选择个体参与进化
        self.neighborSize = np.maximum(population.sizes//10,10)
        self.nr = np.maximum(population.sizes//100,2)
        self.SW = np.zeros((2,self.NIND//2))  # 滑动窗口，可以记录算子的情况
        self.a=0
    
    def reinsertion(self, indices, population, offspring, idealPoint, referPoint):
        
        """
        描述:
            重插入更新种群个体。
        """
        
        weights = referPoint[indices, :]
        pop_ObjV = population.ObjV[indices, :] # 获取邻居个体的目标函数值
        pop_CV = population.CV[indices, :] if population.CV is not None else None # 获取邻居个体的违反约束程度矩阵
        CombinObjV = self.decomposition(pop_ObjV, weights, idealPoint, pop_CV, self.problem.maxormins)
        off_CombinObjV = self.decomposition(offspring.ObjV, weights, idealPoint, offspring.CV, self.problem.maxormins)
        replace = np.where(off_CombinObjV <= CombinObjV)[0][:self.nr]  # 更新个体的索引
        if replace.size == 0:  # 没得替换
            return 
        # 被取代的父代的平均值作为状态
        state = np.mean(population.Chrom[indices[replace]], axis=0)
        state_ = offspring.Chrom[0]
        population[indices[replace]] = offspring                       # 更新子代
        # print("replace:", replace)
        FIR = (CombinObjV[replace] - off_CombinObjV[replace])/CombinObjV[replace]
        r = FIR.sum()
        # print("FIR", FIR)
        # 插入滑动窗口的队列尾
        self.SW = np.concatenate((self.SW[:,1:], np.array([[self.a],[r]])), axis=1)
        # 统计不同算子在滑动窗口里的效果和
        n = 4
        r = np.empty(n)
        for i in range(n):
            r[i] = np.sum(self.SW[1,self.SW[0,:]==i])
            self.DQN.store_transition(state,i,r[i],state_)
        # self.DQN.store_transition(state,self.a,r[self.a],state_)
        
        if self.DQN.memory_counter > 100:
            self.DQN.learn()

    
    def run(self, prophetPop = None): # prophetPop为先知种群（即包含先验知识的种群）
        #==========================初始化配置===========================
        population = self.population
        self.initialization() # 初始化算法模板的一些动态参数
        #===========================准备进化============================
        uniformPoint,NIND = self.uniformPoint, self.NIND
        population.initChrom(NIND)   # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        self.call_aimFunc(population) # 计算种群的目标函数值
        # 生成由所有邻居索引组成的矩阵
        neighborIdx = np.argsort(cdist(uniformPoint, uniformPoint), axis=1, kind='mergesort')[:, :self.neighborSize]
        # 计算理想点
        idealPoint = ea.crtidp(population.ObjV, population.CV, self.problem.maxormins)
        PopCountOpers = []  # 统计不同进化阶段算子选择的结果
        # tcheRange = np.empty(NIND)      # tche距离能差多大
        # meanTcheRange = []
        # CountOpers = np.zeros((self.NIND,self.mutDE.n))  # 统计不同子问题算子选择结果
        #===========================开始进化============================
        while self.terminated(population) == False:
            select_rands = np.random.rand(population.sizes) # 生成一组随机数
            for i in range(population.sizes):
                if select_rands[i] < self.Ps:
                    indices = neighborIdx[i, :] # 得到邻居索引
                else:
                    indices = np.arange(NIND)
                np.random.shuffle(indices)
                offspring = ea.Population(population.Encoding, population.Field, 1) # 实例化一个种群对象用于存储进化的后代（这里只进化生成一个后代）
                # offspring.Chrom = self.mutDE[0].do(population.Chrom, population.Field,i,indices)
                offspring.Chrom,self.a = self.mutDE.do(population.Encoding, population.Chrom, population.Field, i, indices, idealPoint)
                offspring.Chrom = self.mutPolyn.do(offspring.Encoding, offspring.Chrom, offspring.Field) # 变异
                self.call_aimFunc(offspring) # 求进化后个体的目标函数值
                # 更新理想点
                idealPoint = ea.crtidp(offspring.ObjV, offspring.CV, self.problem.maxormins, idealPoint)
                # 重插入更新种群个体
                self.reinsertion(indices, population, offspring, idealPoint, uniformPoint)
                # 画出同一个体在不同进化阶段的选择结果
                # CountOpers[i] += self.mutDE.CountOpers
                # self.mutDE.CountOpers = np.zeros(self.mutDE.n)
                # tcheRange[i] = self.mutDE.TechRange
            # meanTcheRange.append(tcheRange.mean())
            # 每10代记录一次算子选择的结果
            if self.currentGen % 5 == 0:
                PopCountOpers.append(self.mutDE.CountOpers)
                self.mutDE.CountOpers = np.zeros(self.mutDE.n)

        # 画出不同进化阶段算子选择的结果
        BestSelection = np.array(PopCountOpers)
        # 画出不同参考方向算子选择的结果
        # BestSelection = CountOpers
        for i in range(self.mutDE.n):
            plt.plot(BestSelection[:,i],'.',label=self.mutDE.mutOper[i].name)
        plt.legend()
        plt.show()
        # plt.plot(meanTcheRange)
        # plt.show()

        return self.finishing(population),population # 调用finishing完成后续工作并返回结果