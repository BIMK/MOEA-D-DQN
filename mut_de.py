"""
mut_de: 差分算子每次只能接受单个个体
"""
import random
import numpy as np
import geatpy as ea
from Nature_DQN import DQN
from mut import MutM2m

"""
差分变异算子，一次只能变异一个个体
N_ACTIONS = 4  # 动作数量，等于候选算子数量，用于动作选择空间
N_STATES = 30  # 状态维度，等于决策变量维度，用于构造DQN的网络结构
MEMORY_CAPACITY = 100  # 学习MEMORY_CAPCITY个transaction后开始决策
"""


# 将算子的输入输出由种群改成个体
class Random_mut:
    # 随机选择变异算子
    def __init__(self, F=0.5, K=0.6, CR=0.6, DN=1, Loop=False):
        self.name = "Random Selection"
        self.F = F  # 差分变异的缩放因子
        self.K = K  # 应用于差分变异，和缩放因子差不多
        self.CR = CR  # 交叉概率
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 是否采用循环方式处理超出边界的变异结果，用不到
        self.mutOper = [DE_rand_1(), DE_rand_2(), DE_current_to_rand_1(), DE_current_to_rand_2()]

    def do(self, Encoding, OldChrom, FieldDR, r0):
        # r0是基向量索引
        NewChrom = np.zeros_like(OldChrom)
        for i, v in enumerate(r0):
            # 每个个体随机选择变异算子
            mut = self.mutOper[random.randint(0, 3)]
            # mut = self.mutOper[1]
            NewChrom[i] = mut.do(Encoding, OldChrom, FieldDR, v)
        return NewChrom


class Best_mut:
    def __init__(self, Problem, lambda_,F=0.5, K=0.5, CR=1.0, DN=1, Loop=False):
        self.name = "Best Selection"
        self.Problem = Problem
        self.lambda_ = lambda_  # 权重向量
        self.F = F  # 差分变异的缩放因子
        self.K = K  # 应用于差分变异，和缩放因子差不多
        self.CR = CR  # 交叉概率
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 是否采用循环方式处理超出边界的变异结果，用不到
        # self.mutPolyn = ea.Mutpolyn(Pm = 1/self.Problem.Dim, DisI = 20, FixType=4) # 生成多项式变异算子对象
        self.mutOper = [DE_rand_1(F=0.5), DE_rand_2(F=0.5), DE_current_to_rand_2(F=0.5,K=0.5), DE_current_to_rand_1(F=0.5,K=0.5)]
        self.n = len(self.mutOper) # 候选算子个数
        self.CountOpers = np.zeros(self.n)  # 记录算子的选择情况
        self.TechRange = 0

    def do(self, Encoding, OldChrom, FieldDR, r0, indices, z):
        # r0是基向量索引，可以对r0:list做变异
        off = ea.Population(Encoding, FieldDR, self.n) # 实例化一个种群对象用于存储用不同算子进化后的个体
        off.initChrom(self.n)  # 初始化种群染色体矩阵
        # 利用n个算子变异，填充到size=n的种群
        for i in range(self.n):
            off.Chrom[i] = self.mutOper[i].do(OldChrom, FieldDR, r0, indices) # 执行变异
        # off.Chrom = self.mutPolyn.do(off.Encoding, off.Chrom,off.Field)
        # 求off种群的目标函数值
        off.Phen = off.decoding()  # 解码
        self.Problem.aimFunc(off)
        weight = self.lambda_[[r0]*self.n,:]
        tche = ea.tcheby(off.ObjV, weight, z)
        self.TechRange = tche.max() - tche.min()
        # self.TechRange = tche.mean()
        # if np.random.rand() < 0.9:
        # 选择Tech距离最小的作为最优子代
        Techminindex = np.argmin(tche)
        # else:
            # Techminindex = random.sample([0,1,2,3],1)[0]
            # print(Techminindex)
        self.CountOpers[Techminindex]+=1  # 更新算子选择情况
        chrom = np.array([off.Chrom[Techminindex]])
        return chrom


class RL_mut_moea:
    """
    进化算法模板只调用本类，本类负责依据RL，将个体分配给合适的变异算子
    """

    def __init__(self, Problem, lambda_, dqn,F=0.5, K=0.6, CR=0.7, DN=1, Loop=False):
        self.name = "Reinforcement Learning Selection"
        self.Problem = Problem
        self.lambda_ = lambda_  # moead的权重向量
        self.F = F  # 差分变异的缩放因子
        self.K = K  # 应用于差分变异，和缩放因子差不多
        self.CR = CR  # 交叉概率
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 是否采用循环方式处理超出边界的变异结果，用不到
        self.recOpers = [DE_rand_1(), DE_rand_2(),DE_current_to_rand_1(),DE_current_to_rand_2()]
        self.n = len(self.recOpers)
        # self.dqn = DQN(Problem.Dim+Problem.M, self.n)
        # self.dqn = DQN(Problem.Dim, self.n)
        self.dqn = dqn
        self.countOpers = np.zeros(self.n)

    def do(self, Encoding, OldChrom, FieldDR, r0, neighbourVector, z):
        """
        Encoding: 'RI' 编码方式(实数和整数编码)
        OldChrom: 变异前的基因型
        FieldDR:  译码矩阵，对于实整数编码，3行n列，第一行是决策变量下界，第二行是上界
        r0:       差分进化的基向量索引
        """
        s = OldChrom[r0]
        # s = np.ones(30)
        # print(s.shape)
        a = self.dqn.choose_action(s)
        s_ = self.recOpers[a].do(OldChrom, FieldDR, r0, neighbourVector)
        self.countOpers[a] += 1
        return s_,a


        off = ea.Population(Encoding, FieldDR, 2)
        off.initChrom(2)
        off.Chrom[0] = OldChrom[r0]
        off.Chrom[1] = s_[0]
        off.Phen = off.decoding()  # 解码
        self.Problem.aimFunc(off)
        weight = self.lambda_[[r0,r0],:]
        # 计算tech距离
        tche   = ea.tcheby(off.ObjV, weight, z)
        reward = tche[0] - tche[1]

        r = reward
        self.dqn.store_transition(s,a,r,s_[0])
        if self.dqn.memory_counter > 50:
            self.dqn.learn()
        self.CountOpers[a] += 1
        return s_,a



        off = ea.Population(Encoding, FieldDR, 2)
        off.initChrom(2)
        off.Chrom[0] = OldChrom[r0]
        # 状态就是个体的决策向量
        # s = OldChrom[r0]
        s = np.hstack((OldChrom[r0],self.lambda_[r0]))
        # 计算适应度值
        # r = self.Problem.aimIndividualFunc(OldChrom[r0])
        # 计算Tche距离
        # t1 = Tche(self.lambda_[r0], r, z)
        # 根据状态选择变异算子的索引
        a = self.dqn.choose_action(s)
        # a = 0
        # 在环境中执行动作,返回变异后个体Chrom
        s_ = self.mutOper[a].do(OldChrom, FieldDR, r0, neighbourVector)
        off.Chrom[1] = s_
        # NewChrom[i] = s_
        # 计算临时种群的适应度值
        off.Phen = off.decoding()  # 解码
        self.Problem.aimFunc(off)
        weight = self.lambda_[[r0,r0],:]
        # 计算tech距离
        tche   = ea.tcheby(off.ObjV, weight, z)

        # r_ = self.Problem.aimIndividualFunc(s_)
        # t2 = Tche(self.lambda_[r0], r_, z)
        # TODO 目标函数是越小越好，简单的用目标函数值的减少作为reward,但reward的选择仍可以改进
        # Tche也是越小越好
        reward = tche[0] - tche[1]
        # print(reward)
        reward = reward*(10)-8
        reward = -a
        # print(reward-10)
        self.CountOpers[a]+=1 # 记录算子选择情况
        # 存储DQN的决策过程
        self.dqn.store_transition(s, a, reward, np.hstack((s_[0],self.lambda_[r0])))
        if self.dqn.memory_counter > 300:
            # if self.dqn.memory_counter % 10 == 0:
            self.dqn.learn()
        return s_
        # return NewChrom


# class RL_mut:
#     """
#     进化算法模板只调用本类，本类负责依据RL，将个体分配给合适的变异算子
#     """

#     def __init__(self, Problem, F=0.5, K=0.6, CR=0.7, DN=1, Loop=False):
#         self.name = "RL"
#         self.Problem = Problem
#         self.F = F  # 差分变异的缩放因子
#         self.K = K  # 应用于差分变异，和缩放因子差不多
#         self.CR = CR  # 交叉概率
#         self.DN = DN  # 表示有多少组差分向量
#         self.Loop = Loop  # 是否采用循环方式处理超出边界的变异结果，用不到
#         self.dqn = DQN(Problem.Dim, sel)
#         self.mutOper = [DE_rand_1(), DE_rand_2(), DE_current_to_rand_1()]

#     def do(self, Encoding, OldChrom, FieldDR, r0):
#         """
#         Encoding: 'RI' 编码方式(实数和整数编码)
#         OldChrom: 变异前的基因型
#         FieldDR:  译码矩阵，对于实整数编码，3行n列，第一行是决策变量下界，第二行是上界
#         r0:       差分进化的基向量索引
#         """
#         # PopSize, ChromLength = OldChrom.shape
#         NewChrom = np.zeros_like(OldChrom)
#         for i, v in enumerate(r0):
#             # 状态就是个体的决策向量
#             s = OldChrom[i]
#             # 计算目标函数值
#             r = self.Problem.aimIndividualFunc(s)
#             # 根据个体决策变量选择变异算子的索引
#             a = self.dqn.choose_action(s)
#             # 在环境中执行动作,返回变异后个体Chrom
#             s_ = self.mutOper[a].do(Encoding, OldChrom, FieldDR, v)
#             NewChrom[i] = s_
#             # 计算新的适应度值
#             r_ = self.Problem.aimIndividualFunc(s_)
#             # TODO 目标函数是越小越好，简单的用目标函数值的减少作为reward,但reward的选择仍可以改进
#             reward = r - r_
#             # 存储DQN的决策过程
#             self.dqn.store_transition(s, a, reward, s_)
#         if self.dqn.memory_counter > 60:
#             # if self.dqn.memory_counter % 10 == 0:
#             self.dqn.learn()
#         return NewChrom


class DE_rand_1:
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2);
    """

    def __init__(self, F=0.5, CR=0.3, DN=1, Loop=False):
        self.name = "DE_rand_1"
        self.F = F  # 差分变异缩放因子
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 表示是否采用循环的方式处理超出边界的变异结果
        self.CR = CR  # 交叉概率

    def do(self, OldChrom, FieldDR, r0, neighbourVector):  # 执行变异
        """
        对OldChrom中第r0个个体进行变异
        OldChrom: 种群基因型
        neighbourVector: 邻域个体，r0是待变异个体，r1 r2从neighbourVector中选择
        """
        # vi = xi + F × (xr1 − xr2 );
        # TODO xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        # PopSize, ChromLength = OldChrom.shape
        # print(PopSize, ChromLength)
        r1, r2 = neighbourVector[0], neighbourVector[1]
        # 变异
        x = OldChrom[r0]
        v = x + self.F * (OldChrom[r1] - OldChrom[r2])
        # 边界处理
        v = processBound(v,FieldDR)
        return v

    def getHelp(self):  # 查看内核中的变异算子的API文档
        # help(mutde)
        print("比如")


class DE_rand_2:
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2 + xr3 - xr4);
    """

    def __init__(self, F=0.5, CR=0.3, DN=1, Loop=False):
        self.name = 'DE_rand_2'
        self.F = F  # 差分变异缩放因子
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 表示是否采用循环的方式处理超出边界的变异结果
        self.CR = CR

    def do(self, OldChrom, FieldDR, r0, neighbourVector):  # 执行变异
        PopSize, ChromLength = OldChrom.shape
        # vi = xi + F × (xr1 − xr2 +xr3 -xr4);
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0]
        r1, r2, r3, r4 = neighbourVector[0], neighbourVector[1], neighbourVector[2], neighbourVector[3]
        v = x + self.F * (OldChrom[r1] - OldChrom[r2] + OldChrom[r3] - OldChrom[r4])
        # 边界处理
        v = processBound(v, FieldDR)
        return v

    def getHelp(self):  # 查看内核中的变异算子的API文档
        # help(mutde)
        print("比如")


class DE_current_to_rand_1:
    # vi=xi+K(xi-xr1)+F(xr2-xr3)
    def __init__(self, F=0.5, K=0.6):
        self.F = F
        self.K = K
        self.name = 'DE_current_rand_1'

    def do(self, OldChrom, FieldDR, r0, neighbourVector):  # 执行变异
        """
        r0:基向量索引
        FieldDR：译码矩阵 [lb:变量下界;ub:变量上界;varTypes:0:连续1:离散]
        """
        PopSize, ChromLength = OldChrom.shape
        # vi=xi+K(xi-xr1)+F(xr2-xr3)
        r1, r2, r3 = neighbourVector[0], neighbourVector[1], neighbourVector[2]
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0]
        v = x + self.K * (x - OldChrom[r1]) + self.F * (OldChrom[r2] - OldChrom[r3])
        # 边界处理
        v = processBound(v, FieldDR)
        return v

    def getHelp(self):
        pass


class DE_current_to_rand_2:
    # v1=xi+K(xi-xr1)+F(xr2-xr3)+F(xr4-xr5)
    def __init__(self, F=0.5, K=0.6):
        self.F = F
        self.K = K
        self.name = 'DE_current_rand_2'

    def do(self, OldChrom, FieldDR, r0, neighbourVector):  # 执行变异
        """
        xi:基向量索引
        FieldDR：译码矩阵 [lb:变量下界;ub:变量上界;varTypes:0:连续1:离散]
        """
        PopSize, ChromLength = OldChrom.shape
        r1, r2, r3, r4, r5 = neighbourVector[0], neighbourVector[1], neighbourVector[2], neighbourVector[3], neighbourVector[4]
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0]
        xr1 = OldChrom[r1]
        xr2 = OldChrom[r2]
        xr3 = OldChrom[r3]
        xr4 = OldChrom[r4]
        xr5 = OldChrom[r5]
        v = x + self.K * (x - xr1) + self.F * (xr2 - xr3 + xr4 - xr5)
        # 边界处理
        v = processBound(v, FieldDR)
        return v


def processBound(v, FieldDR):
    # 边界处理
    lb = FieldDR[0]
    ub = FieldDR[1]
    v[v < lb] = lb[v < lb]
    v[v > ub] = ub[v > ub]
    v = np.array([v]) # 增加一维，shape=(1,ChromLength)
    return v


def PolynMut(v, FieldDR):
    lb = FieldDR[0]
    ub = FieldDR[1]
    # 边界处理
    v = np.minimum(np.maximum(v,lb),ub)
    v = np.array([v]) # 增加一维，shape=(1,ChromLength)
    N,D = v.shape
    mu = np.random.rand(D)
    
