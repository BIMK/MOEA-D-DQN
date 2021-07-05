"""
交叉算子池，交叉算子统一不处理边界信息
1. 模拟二进制交叉
2. 差分算子
3. MOEA/D-M2M里的交叉
"""
import numpy as np
import geatpy as ea
from Nature_DQN import DQN
import random


class Random_cro:
    # 随机选择交叉算子
    def __init__(self, maxgen, FieldDR, Encoding):
        self.name = "Random Selection"
        self.FieldDR = FieldDR
        self.Encoding = Encoding
        # self.Loop = Loop  # 是否采用循环方式处理超出边界的变异结果，用不到
        self.Opers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(maxgen), DE_rand_1(), DE_rand_2(),
                      DE_current_to_rand_1(), DE_current_to_rand_2()]
        self.n = len(self.Opers)
        self.processBound = ProcessBound(FieldDR)

    def do(self, OldChrom, r0, neighbourVector, currentGen):
        # off = ea.Population(self.Encoding, self.FieldDR, 1)
        idx = random.randint(0, self.n)
        if idx == 0:
            Chrom = self.Opers[0].do(OldChrom, r0, neighbourVector)
        elif idx == 1:
            Chrom = self.Opers[1].do(OldChrom, r0, neighbourVector, currentGen)
        else:
            Chrom = self.Opers[idx].do(OldChrom, r0, neighbourVector)
        return Chrom


class Best_cro:
    def __init__(self, Problem, lambda_, maxgen, Encoding, FieldDR):
        self.name = "Best Selection"
        self.Problem = Problem
        self.lambda_ = lambda_  # 权重向量
        self.Encoding = Encoding
        self.FieldDR = FieldDR
        # self.Opers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(maxgen), DE_rand_1(), DE_rand_2(),
        #   DE_current_to_rand_1(), DE_current_to_rand_2()]
        self.Opers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(maxgen), DE_rand_1(), DE_rand_2()]
        self.n = len(self.Opers)  # 候选算子个数
        self.countOpers = np.zeros(self.n)  # 记录算子的选择情况
        self.processBound = ProcessBound(FieldDR)

    def do(self, OldChrom, r0, neighbourVector, z, currentGen):
        # r0是基向量索引，可以对r0:list做变异
        off = ea.Population(self.Encoding, self.FieldDR, self.n)  # 实例化一个种群对象用于存储用不同算子进化后的个体
        off.initChrom(self.n)  # 初始化种群染色体矩阵
        # 使用n个交叉算子，填充到size=n的种群
        # 模拟二进制交叉和M2m里的交叉需要单独处理
        off.Chrom[0] = self.Opers[0].do(OldChrom, r0, neighbourVector)
        off.Chrom[1] = self.Opers[1].do(OldChrom, r0, neighbourVector, currentGen)
        for i in range(2, self.n):
            off.Chrom[i] = self.Opers[i].do(OldChrom, r0, neighbourVector)  # 执行变异
        # 既然要求目标函数值，就需要处理边界
        off.Chrom = self.processBound.do(off.Chrom)
        # self.processBound.do(off.Chrom)  # 处理边界

        off.Phen = off.decoding()  # 解码
        self.Problem.aimFunc(off)
        weight = self.lambda_[[r0] * self.n, :]
        tche = ea.tcheby(off.ObjV, weight, z)
        # self.TechRange = tche.max() - tche.min()
        # self.TechRange = tche.mean()
        # 选择Tech距离最小的作为最优子代
        Techminindex = np.argmin(tche)
        # print(Techminindex)
        self.countOpers[Techminindex] += 1  # 更新算子选择情况
        chrom = np.array([off.Chrom[Techminindex]])
        # 确实比应用单个算子效果好
        # chrom = np.array([off.Chrom[3]])
        return chrom


class RecRL:
    """
    用强化学习选择交叉算子
    """

    def __init__(self, problem, lambda_, maxgen, NIND) -> None:
        self.name = "RecRL"
        self.problem = problem
        self.lambda_ = lambda_
        # self.recOpers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(maxgen), DE_rand_1(), DE_rand_2(),
        #  DE_current_to_rand_1(), DE_current_to_rand_2()]
        self.Opers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(maxgen), DE_rand_1(), DE_rand_2()]
        self.n = len(self.Opers)
        self.dqn = DQN(problem.Dim + problem.M, self.n)
        self.SW = np.zeros((2, NIND * 4))
        self.a = 0
        self.state = None
        self.state_ = None
        self.countOpers = np.zeros(self.n)  # 统计不同算子的选择频率

    def do(self, OldChrom, r0, neighbourVector, currentGen):
        """
        r0: 父代在OldChrom里的索引
        OldChrom: 变异前的原始矩阵
        return:  返回新种群的染色体矩阵
        """
        self.gen = currentGen
        self.state = np.hstack((OldChrom[r0], self.lambda_[r0]))
        if self.dqn.memory_counter > 300:
            self.a = self.dqn.choose_action(self.state)
            # 如果有算子在滑动窗口里没有记录，那么就选它
            for i in range(self.n):
                if np.sum(self.SW[0] == i) == 0:
                    self.a = i
                    break
        else:
            self.a = np.random.randint(0, self.n)
        # self.a = np.random.randint(0, self.n)
        # self.a = 0
        self.countOpers[self.a] += 1
        if self.a == 0:  # 使用模拟二进制交叉
            offChrom = self.Opers[0].do(OldChrom, r0, neighbourVector)
        elif self.a == 1:  # 使用M2m里的交叉
            offChrom = self.Opers[1].do(OldChrom, r0, neighbourVector, currentGen)
        else:  # 使用差分算子
            offChrom = self.Opers[self.a].do(OldChrom, r0, neighbourVector)
        self.state_ = np.hstack((offChrom[0], self.lambda_[r0]))
        return offChrom

    def learn(self, r):
        """
        更新DQN
        :param r: 子代相对于父代适应度的提高率
        """
        # 将上次进化加入滑动窗口
        self.SW = np.concatenate((self.SW[:, 1:], np.array([[self.a], [r]])), axis=1)

        # r1 = np.empty(self.n)
        # r2 = np.empty(self.n)
        # r3 = np.empty(self.n)
        # for i in range(self.n):
        #     r1[i] = np.mean(self.SW[1, self.SW[0, :] == i])
        #     r2[i] = np.median(self.SW[1, self.SW[0, :] == i])
        #     if len(self.SW[1, self.SW[0, :] == i]):
        #         r3[i] = np.max(self.SW[1, self.SW[0, :] == i])
        #     else:
        #         r3[i] = 0
        # print("gen", self.gen)
        # print("mean", r1)
        # print("median", r2)
        # print("max", r3)

        # self.DQN.store_transition(state,i,r[i],state_)
        # reward = np.sum(self.SW[1, self.SW[0, :] == self.a])
        reward = np.max(self.SW[1, self.SW[0, :] == self.a])
        # reward = 6 - self.a
        # print(reward1, reward)
        self.dqn.store_transition(self.state, self.a, reward, self.state_)
        # 学习,更新DQN
        if self.dqn.memory_counter > 200:
            self.dqn.learn()


"""
模拟二进制交叉 Simulated Binary Crossover  geatpy有实现

NewChrom =  recsbx(OldChrom, XOVR, Half, n, Parallel)
        该函数把输入的OldChrom种群染色体矩阵进行模拟二进制交叉，并返回新的种群染色体矩阵。
        交配的一对是有序的，种群的前一半个体和后一半个体进行配对。
        若个体数是奇数，则最后一个个体不参与配对。

OldChrom: 待交叉种群染色体矩阵
XOVR:     交叉概率，默认0.7
Half:     默认False，当为True时，每对交叉结果只选择第一个保留。
n   :     分布指数，默认20，越大交叉结果越接近双亲
Parallel: False
"""


class Recsbx:
    def __init__(self, XOVR=0.9, Half=True, n=20, Parallel=False):
        self.name = "Recsbx"
        self.XOVR = XOVR
        self.Half = Half
        self.n = n
        self.Parallel = Parallel

    def do(self, OldChrom, r0, neighbourVector):
        # pop[r0]作为父代,还需要另外一个父代才能做交叉,所以需要从neighbourVector里随机选一个
        r1 = np.random.choice(neighbourVector)
        p0 = OldChrom[r0:r0 + 1]
        p1 = OldChrom[r1:r1 + 1]
        # print(p1)
        D = p1.shape[1]
        mu = np.random.rand(1, D)
        beta = np.zeros((1, D))
        idx = mu <= 0.5
        beta[idx] = (2 * mu[idx])**(1 / (self.n + 1))
        beta[~idx] = (2 - 2 * mu[~idx])**(-1 / (self.n + 1))
        beta = beta * np.random.choice([-1, 1], (1, D))
        idx = np.random.rand(1, D) < 0.5
        beta[idx] = 1
        # if random.random() < 0.5:
        # beta = 1
        if random.random() < 0.5:
            off = (p0 + p1) / 2 + beta * (p0 - p1) / 2
        else:
            off = (p0 + p1) / 2 - beta * (p0 - p1) / 2
        # off = np.array([off])
        # print(off)
        # r1 = np.random.choice(neighbourVector)
        # p1 = np.vstack((OldChrom[r0], OldChrom[r1]))
        # off = ea.recsbx(p1, self.XOVR, self.Half, self.n, self.Parallel)
        return off


class RecM2m:
    """
    MOEA/D-M2M里的交叉
    """

    def __init__(self, maxgen: int) -> None:
        self.name = 'RecM2m'
        self.MaxGen = maxgen

    def do(self, OldChrom, r0, neighbourVector, currentGen: int):
        N, D = OldChrom.shape
        # r1, r2 = np.random.choice(neighbourVector, 2, replace=False)  # 不放回的抽取2个
        r2 = np.random.choice(neighbourVector, 1, replace=False)  # 不放回的抽取
        p1 = OldChrom[r0]
        p2 = OldChrom[r2]
        # D  = len(p1)   #  决策变量维度
        rc = (2 * np.random.rand(1) - 1) * (1 - np.random.rand(1) ** (-(1 - currentGen / (self.MaxGen + N)) ** 0.7))
        # print(currentGen, self.MaxGen)
        OffDec = p1 + rc * (p1 - p2)
        return OffDec


class DE_rand_1:
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2);
    """

    def __init__(self, F=0.5):
        self.name = "DE_rand_1"
        self.F = F  # 差分变异缩放因子

    def do(self, OldChrom, r0, neighbourVector):  # 执行变异
        """
        对OldChrom中第r0个个体进行变异
        OldChrom: 种群基因型
        neighbourVector: 邻域个体，r0是待变异个体，r1 r2从neighbourVector中选择
        return: v.shape = 1xD
        """
        r1, r2 = neighbourVector[0], neighbourVector[1]
        # 变异
        x = OldChrom[r0:r0 + 1, :]  # 保持x是二维
        v = x + self.F * (OldChrom[r1] - OldChrom[r2])
        return v


class DE_rand_2:
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2 + xr3 - xr4);
    """

    def __init__(self, F=0.5):
        self.name = 'DE_rand_2'
        self.F = F  # 差分变异缩放因子

    def do(self, OldChrom, r0, neighbourVector):  # 执行变异
        # vi = xi + F × (xr1 − xr2 +xr3 -xr4);
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0:r0 + 1, :]
        r1, r2, r3, r4 = neighbourVector[0], neighbourVector[1], neighbourVector[2], neighbourVector[3]
        v = x + self.F * (OldChrom[r1] - OldChrom[r2] + OldChrom[r3] - OldChrom[r4])
        return v


class DE_current_to_rand_1():
    """
    vi=xi+K(xi-xr1)+F(xr2-xr3)
    """

    def __init__(self, F=0.5, K=0.6):
        self.F = F
        self.K = K
        self.name = 'DE_current_rand_1'

    def do(self, OldChrom, r0, neighbourVector):  # 执行变异
        # vi=xi+K(xi-xr1)+F(xr2-xr3)
        r1, r2, r3 = neighbourVector[0], neighbourVector[1], neighbourVector[2]
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0:r0 + 1, :]
        v = x + self.K * (x - OldChrom[r1]) + self.F * (OldChrom[r2] - OldChrom[r3])
        return v


class DE_current_to_rand_2():
    """
    # v1=xi+K(xi-xr1)+F(xr2-xr3)+F(xr4-xr5)
    """

    def __init__(self, F=0.5, K=0.6):
        self.F = F
        self.K = K
        self.name = 'DE_current_rand_2'

    def do(self, OldChrom, r0, neighbourVector):  # 执行变异
        r1, r2, r3, r4, r5 = neighbourVector[0], neighbourVector[1], neighbourVector[2], neighbourVector[3], neighbourVector[4]
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        x = OldChrom[r0:r0 + 1, :]
        xr1 = OldChrom[r1]
        xr2 = OldChrom[r2]
        xr3 = OldChrom[r3]
        xr4 = OldChrom[r4]
        xr5 = OldChrom[r5]
        v = x + self.K * (x - xr1) + self.F * (xr2 - xr3 + xr4 - xr5)
        return v


class ProcessBound:
    """
    空操作，仅处理边界信息
    """

    def __init__(self, FieldDR) -> None:
        self.FieldDR = FieldDR

    def do(self, OldChrom):
        lb = self.FieldDR[0]
        ub = self.FieldDR[1]
        # OffChrom = OldChrom.copy()
        # 边界处理
        OffChrom = np.minimum(np.maximum(OldChrom, lb), ub)
        return OffChrom


class RecDirect:
    """
    交叉和变异各有一个空操作，变异的空操作会处理边界，交叉的空操作直接返回原种群信息
    但是我怀疑空操作会降低算法的性能
    """

    def do(self, OldChrom):
        return OldChrom


if __name__ == '__main__':
    # 检查RecM2m
    # p1 = np.array([1,2,3])
    # p2 = np.array([2,1,4])
    # randd = np.array([0.3])
    # currentGen = 6
    # MaxGen = 13
    # rc = (2*randd-1) * (1-randd ** (-(1-currentGen/MaxGen)**0.7))
    # OffDec = p1 + rc*(p1-p2)
    # print(OffDec)

    Chrom = np.array([[1, 2, 3], [4, 5, 6]])
    rec = ea.Recsbx(Half=True)
    off = rec.do(Chrom)
    print(off)
