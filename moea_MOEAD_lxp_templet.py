# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from matplotlib import pyplot as plt
from moead_utils import *
from mut_de import DE_rand_1, DE_rand_2, DE_current_to_rand_1, DE_current_to_rand_2, Best_mut
import random
import sys


class moea_MOEAD_templet(ea.MoeaAlgorithm):
    """
    moea_MOEAD_templet: class - MOEA\D算法模版

模板使用注意:
    本模板调用的目标函数形如：aimFunc(pop),
    其中pop为Population类的对象，代表一个种群，
    pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
    该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
    若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
    该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中，
                          违反约束程度矩阵保存在种群对象的CV属性中。
    例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
         此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
    若不符合上述规范，则请修改算法模板或自定义新算法模板。
    """

    def __init__(self, problem, population, T=11, delta=0.9):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = "MOEA-D"
        self.T = T  # 邻域个数
        self.z = None
        self.nr = 2  # 至多更新邻域中的两个
        self.NIND = population.sizes  # 种群中个体数量
        self.delta = delta  # 父代有delta概率从邻域中选择
        self.lambda_ = initWeightVector(self.NIND)  # 生成权重向量
        self.neighbourVector = initNeighbourVector(self.lambda_, T)  # 生成邻域关系
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        # self.mutOper = DE_rand_1(F=0.5)  # 变异算子，自带交叉操作
        # self.mutOper = DE_rand_2(F=1.25)
        # self.mutOper = DE_current_to_rand_1(F=1.2, K=0.2)
        # self.mutOper = DE_current_to_rand_2(F=1.2, K=0.2)
        # self.mutOper = RL_mut_moea(problem, self.lambda_)
        self.mutOper = Best_mut(problem, self.lambda_)
        self.mutOper.F = 0.5
        self.mutPol = ea.Mutpolyn(Pm=0.01)
    def run(self):
        # =======================初始化配置==================================
        population = self.population
        self.initialization()  # 初始化MOEA模版的动态参数
        # ========================准备进化==================================
        # if population.Chrom is None:
        population.initChrom(self.NIND)  # 初始化种群染色体矩阵,详见Population的源码，内含解码
        # else:
        population.Phen = population.decoding()  # 染色体解码
        self.problem.aimFunc(population)  # 计算种群目标函数值
        self.z = population.ObjV.min(axis=0, keepdims=True)# 当前目标函数的最佳值 shape=(1,2)
        self.evalsNum = 0  # 记录评价次数
        self.PopCountOpers = [] # 记录算子的选择结果
        self.CountOpers = np.zeros((self.NIND,self.mutOper.n))
        # ======================开始进化=============================
        while self.terminated(population) == False:
            
            self.ter = 0  # 记录每次种群更新次数
            # print("currentGen = {}".format(self.z))
            # 第i个权重向量的邻域中随机选择两个权重向量对应的个体进行交叉变异产生新的个体
            select_rands = np.random.rand(population.sizes) # 生成一组随机数
            for i in np.random.permutation(self.NIND):
            # for i in range(self.NIND):
                if select_rands[i] < self.delta:
                    indices = self.neighbourVector[i]  # 父代只从邻域中选择
                else:
                    indices = np.arange(self.NIND)  # 父代从整个种群中选择
                np.random.shuffle(indices)
                # 变异交叉产生新的个体
                # 差分变异算子会从邻域中随机选择父代个体
                offspring = ea.Population(population.Encoding, population.Field, 1) # 实例化一个种群对象用于存储进化的后代
                # 强化学习算法需要多加一个z参数
                offspring.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field, i, indices, self.z)
                # offspring.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field, i, indices)
                
                offspring.Chrom = np.array([offspring.Chrom])
                offspring.Phen = offspring.decoding()  # 解码
                self.problem.aimFunc(offspring) # 计算目标函数值
                # 更新z，每个目标的最优值
                self.z = np.vstack((self.z, offspring.ObjV))
                self.z = self.z.min(axis=0, keepdims=True)
                # print("第{0:3d}代第{1:3d}个体".format(self.currentGen,i))
                # print(offspring.ObjV)
                # print(self.z)
                # 按照Tchebycheff分解法确定是否需要更新邻域,更新邻域中所有比offspring差的个体
                # TODO Tche一次算出所有个体
                c = 0
                for j in indices:
                    if c >= self.nr:
                        break
                    t1 = Tche(self.lambda_[j], offspring.ObjV[0], self.z[0])
                    t2 = Tche(self.lambda_[j], population.ObjV[j], self.z[0])

                    if t1 <= t2:  # 更新邻域
                        population.Chrom[j] = offspring.Chrom[0]
                        population.Phen[j] = offspring.Phen[0]
                        population.ObjV[j] = offspring.ObjV[0]
                        self.ter += 1
                        c += 1
                self.evalsNum += 1  # 更新评价次数
                # print(self.currentGen,end=' ')
            # 记录进化不同阶段算子选择结果,每10代记录一次
            # if (self.currentGen) % 10 == 0: 
            #     self.PopCountOpers.append(self.mutOper.CountOpers)
            #     self.mutOper.CountOpers = np.zeros(self.mutOper.n)
                # print("currentGen = {}  ".format(self.currentGen+1))
                # print(self.mutOper.CountOpers, sum(self.mutOper.CountOpers))
                self.CountOpers[i]+=self.mutOper.CountOpers
                self.mutOper.CountOpers = np.zeros(self.mutOper.n)
            self.mutPol.do(population.Encoding, population.Chrom[1:2], population.Field)
        # print(self.currentGen)
        # print(self.mutOper.CountOpers)  # best mut 算子选择情况
        # for i in range(self.NIND):
            # print(self.CountOpers[i], sum(self.CountOpers[i]))

        """按照不同个体画出算子选择结果"""
        plt.plot(self.CountOpers[:,0])
        plt.plot(self.CountOpers[:,1])
        plt.plot(self.CountOpers[:,2])
        plt.plot(self.CountOpers[:,3])
        plt.show()


        """按照进化阶段画出算子选择的结果"""
        # self.PopCountOpers = np.array(self.PopCountOpers)
        # plt.plot(self.PopCountOpers[:,0],'.')
        # plt.plot(self.PopCountOpers[:,1],'.')
        # plt.plot(self.PopCountOpers[:,2],'.')
        # plt.plot(self.PopCountOpers[:,3],'.')
        # plt.show()


        # ea.moeaplot(population.ObjV, self.mutOper.name, saveFlag=False, gridFlag=True)
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
