# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import geatpy as ea  # 导入geatpy库
from scipy.spatial.distance import cdist
from sys import path as paths
from os import path
from matplotlib import pyplot as plt
from Nature_DQN import DQN
from mut_de import DE_rand_1, DE_rand_2, DE_current_to_rand_1, DE_current_to_rand_2, RL_mut_moea
from crossover import RecRL

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class moea_MOEAD_DRA_templet(ea.MoeaAlgorithm):
    
    def __init__(self, problem, population, MAXGEN):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'MOEA/D-DRA-DE'
        self.MAXGEN = MAXGEN
        self.uniformPoint, self.NIND = ea.crtup(self.problem.M, population.sizes)  # 生成在单位目标维度上均匀分布的参考点集
        self.DQN = DQN(problem.Dim, 4)
        if population.Encoding == 'RI':
            # self.mutDE = [DE_rand_1(),DE_rand_2(),DE_current_to_rand_1(),DE_current_to_rand_2()]
            # self.mutOper = RL_mut_moea(problem, self.uniformPoint, self.DQN)
            self.mutOper = RecRL(problem, self.uniformPoint, MAXGEN, self.NIND)
            self.mutPolyn = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20, FixType=4)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''RI''.')
        if self.problem.M <= 2:
            self.decomposition = ea.tcheby  # 采用切比雪夫权重聚合法
        else:
            self.decomposition = ea.pbi  # 采用pbi权重聚合法
        
        self.Ps = 0.9  # (Probability of Selection)表示进化时有多大的概率只从邻域中选择个体参与进化
        self.neighborSize = max(self.NIND // 10, 20)
        self.nr = max(self.NIND // 100, 3)
        # self.SW = np.zeros((2, self.NIND // 2))  # 滑动窗口，可以记录算子的情况
        # self.a = 0
    
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
    
    def reinsertion(self, indices, population, offspring, idealPoint, referPoint, i):
        """
        描述:
            重插入更新种群个体。
            indices: 父代池，里面的个体可以被替换
            i: 父代，它的决策变量应该作为RL的state
            idealPoint: 理想点，每个目标上的最优点
            referPoint: 参考点，权重向量
        """
        
        weights = referPoint[indices, :]
        pop_ObjV = population.ObjV[indices, :]  # 获取邻居个体的目标函数值
        # 获取邻居个体的违反约束程度矩阵
        pop_CV = population.CV[indices, :] if population.CV is not None else None
        CombinObjV = self.decomposition(pop_ObjV, weights, idealPoint, pop_CV, self.problem.maxormins)
        off_CombinObjV = self.decomposition(offspring.ObjV, weights, idealPoint, offspring.CV, self.problem.maxormins)
        replace = np.where(off_CombinObjV <= CombinObjV)[0][:self.nr]  # 更新个体的索引
        population[indices[replace]] = offspring  # 更新子代
        # if replace.size == 0:  # 没得替换
        # return
        # 被取代的父代的平均值作为状态
        # 直系父代的决策变量作为state
        # state = np.mean(population.Chrom[indices[replace]], axis=0)
        # state = population.Chrom[i]
        # state_ = offspring.Chrom[0]
        if not isinstance(self.mutOper, RecRL):
            return
        # 子代相比父代适应度提高的相对率
        FIR = (CombinObjV[replace] - off_CombinObjV[replace]) / CombinObjV[replace]
        r = FIR.sum()
        self.mutOper.learn(r)
        # # 插入滑动窗口的队列尾
        # self.SW = np.concatenate((self.SW[:, 1:], np.array([[self.a], [r]])), axis=1)
        # # 统计不同算子在滑动窗口里的reward sum
        # n = 4  # 算子池里的算子数量
        # r = np.empty(n)
        # for i in range(n):
        #     r[i] = np.sum(self.SW[1, self.SW[0, :] == i])
        #     # self.DQN.store_transition(state,i,r[i],state_)
        # self.DQN.store_transition(state, self.a, r[self.a], state_)
        #
        # if self.DQN.memory_counter > 100:  # 算子池里累积一定数量经验再学习
        #     self.DQN.learn()
        #
    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        self.initialization()
        # NOTE: 在使用crtup生成单位目标维度均匀分布的参考点集时NIND可能不是种群大小。
        # NOTE: 为了保证评价次数不变，要更改MAXGEN
        self.MAXGEN = round((self.MAXGEN * population.sizes) / self.NIND + 0.5)
        # ===========================准备进化============================
        uniformPoint, NIND = self.uniformPoint, self.NIND
        # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        population.initChrom(NIND)
        self.call_aimFunc(population)  # 计算种群的目标函数值
        # 生成由所有邻居索引组成的矩阵
        neighborIdx = np.argsort(cdist(uniformPoint, uniformPoint), axis=1, kind='mergesort')[:, :self.neighborSize]
        # 计算理想点
        idealPoint = ea.crtidp(population.ObjV, population.CV, self.problem.maxormins)
        pi = np.ones((NIND, 1))  # 初始化pi作为子问题选择基准
        # old Techbycheff function value of each solution on its subproblem 当前每个子问题的目标值
        # oldz = ea.tcheby(population.ObjV,uniformPoint,idealPoint)
        oldz = self.decomposition(population.ObjV, uniformPoint, idealPoint)
        PopCountOpers = []  # 统计不同进化阶段算子选择的结果
        # CountOpers = np.zeros((self.NIND,self.mutDE.n))  # 统计不同子问题算子选择结果
        # ===========================开始进化============================
        while self.terminated(population) == False:
            for _ in range(5):
                # 这里的边界是指weight在M-1个方向上权重为0
                Bounday = np.where(np.sum(self.uniformPoint < 0.0001, 1) == (self.problem.M - 1))[0].tolist()
                # 利用10元竞赛选出pi最大的N/5-problem.M个体的索引
                I = self.tournamentSelection(10, NIND // 5 - self.problem.M, pi) + Bounday
                select_rands = np.random.rand(NIND)  # 生成一组随机数
                for i in I:
                    if select_rands[i] < self.Ps:
                        indices = neighborIdx[i, :]  # Ps的概率从邻域中选
                    else:
                        indices = np.arange(NIND)
                    np.random.shuffle(indices)
                    # 实例化一个种群对象用于存储进化的后代（这里只进化生成一个后代）
                    offspring = ea.Population(population.Encoding, population.Field, 1)
                    # offspring.Chrom = self.mutDE[0].do(population.Chrom, population.Field, i, indices)
                    # offspring.Chrom, self.a = self.mutOper.do(population.Encoding, population.Chrom, population.Field, i, indices, idealPoint)
                    offspring.Chrom = self.mutOper.do(population.Chrom, i, indices, self.currentGen)
                    offspring.Chrom = self.mutPolyn.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
                    self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
                    # 更新理想点
                    idealPoint = ea.crtidp(offspring.ObjV, offspring.CV, self.problem.maxormins, idealPoint)
                    # 重插入更新种群个体
                    self.reinsertion(indices, population, offspring, idealPoint, uniformPoint, i)
            """每10代更新一次pi值"""
            if self.currentGen % 10 == 0:
                newz = ea.tcheby(population.ObjV, uniformPoint, idealPoint)
                DELTA = (oldz - newz) / oldz
                idx = DELTA < 0.001
                pi[~idx] = 1
                pi[idx] = (0.95 + 0.05 * DELTA[idx] / 0.001) * pi[idx]
                oldz = newz
                """统计不同进化阶段算子选择的结果"""
                PopCountOpers.append(self.mutOper.countOpers)
                self.mutOper.countOpers = np.zeros(self.mutOper.n)  # 清空算子选择记录器
        
        # 画出不同进化阶段算子选择的结果
        BestSelection = np.array(PopCountOpers)
        # 画出不同子问题算子选择的结果
        # BestSelection = CountOpers
        # matplotlib.use('agg')
        # for i in range(self.mutOper.n):
        #     plt.plot(BestSelection[:,i],'.',label=self.mutOper.recOpers[i].name)
        # plt.legend()
        # plt.show()
        return self.finishing(population), population, plt  # 调用finishing完成后续工作并返回结果
