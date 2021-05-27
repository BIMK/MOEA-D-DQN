"""
变异算子池，包括
1. 多项式变异
2. 高斯变异
3. MOEA/D-M2M的变异

应当先交叉再变异，因为变异算子会检查边界值

为了和geatpy库兼容，传入的仍旧是一个种群，只是种群里的个体只有一个。
"""
import numpy as np
import geatpy as ea
from Nature_DQN import DQN


class MutRL:
    """
    使用强化学习做算子选择,需要有一个滑动窗口shape=(2,N//2)设计为种群的一半大小,存储应用的算子和得到的improvement.
    """

    def __init__(self, problem, lambda_, Encoding, FieldDR, maxgen, NIND) -> None:
        self.name = "MutRL"
        self.problem = problem
        self.lambda_ = lambda_
        self.Encoding = Encoding
        self.FieldDR = FieldDR
        self.mutOpers = [Mutpolyn(self.Encoding, self.FieldDR), Mutgau(self.Encoding, self.FieldDR), MutM2m(self.FieldDR, maxgen)]
        self.n = len(self.mutOpers)
        self.dqn = DQN(problem.Dim + problem.M, self.n)
        self.SW = np.zeros((2, NIND // 2))
        self.a = 0
        self.state = None
        self.state_ = None
        self.countOpers = np.zeros(self.n)

    def do(self, OldChrom, Parent1, r0, currentGen):
        self.state = np.hstack((Parent1[0], self.lambda_[r0]))
        if self.dqn.memory_counter > 100:
            self.a = self.dqn.choose_action(self.state)
            # 优先选择在SW中没有记录的算子
            for i in range(self.n):
                if np.sum(self.SW[0] == i) == 0:
                    self.a = i
                    break
        else:
            self.a = np.random.randint(0, self.n)
        self.countOpers[self.a] += 1
        if self.a == 0 or self.a == 1:
            OffChrom = self.mutOpers[self.a].do(Parent1)
        elif self.a == 2:  # mutM2m
            OffChrom = self.mutOpers[self.a].do(OldChrom, r0, Parent1, currentGen)
        else:
            raise Exception("self.a 选择超范围")
        self.state_ = np.hstack((OffChrom[0], self.lambda_[r0]))
        # print(self.a, OffChrom.shape)
        return OffChrom

    def learn(self, r):
        """
        更新DQN
        :param r: 子代的适应度提高
        """
        self.SW = np.concatenate((self.SW[:, 1:], np.array([[self.a], [r]])), axis=1)
        reward = np.sum(self.SW[1, self.SW[0, :] == self.a])
        self.dqn.store_transition(self.state, self.a, reward, self.state_)
        # 学习,更新DQN
        if self.dqn.memory_counter > 100:
            self.dqn.learn()


"""
多项式变异，geatpy已经提供

NewChrom = mutpolyn(Encoding, OldChrom, FieldDR, Pm, DisI, FixType, Parallel)
        该函数让一个实整数编码种群（Encoding为'RI'）染色体矩阵根据其突变率让个体的每个决策变量发生多项式突变，
        返回一个新的种群染色体矩阵。该函数先是采用实数值来进行变异，
        当结果超出译码矩阵FieldDR所设定的范围时，该函数会对其进行修复，支持截断、循环、往复、随机4种修复方式：

Encoding: 染色体编码方式-RI
OldChrom: 种群染色体矩阵
FieldDR:  指明每个变量的上下界
Pm:       变异算子所发生作用的最小片段发生变异的概率
FixType:  用某种方式对超出范围的染色体进行修复
"""


class Mutpolyn:
    def __init__(self, Encoding, FieldDR, Pm=None, DisI=20, FixType=3, Parallel=False):
        self.Encoding = Encoding
        self.FieldDR = FieldDR
        self.Pm = Pm
        self.DisI = DisI
        self.FixType = FixType
        self.Parallel = Parallel

    def do(self, Parent1):
        Offspring = ea.mutpolyn(self.Encoding, Parent1, self.FieldDR, self.Pm, self.DisI, self.FixType)
        return Offspring


"""
高斯变异，geatpy已经提供

NewChrom = mutgau(Encoding, OldChrom, FieldDR, Pm, Sigma3, Middle, FixType, Parallel)
        该函数对一个实整数编码种群染色体矩阵（Encoding为'RI'）根据其突变率让个体的每个决策变量发生高斯变异，
        并返回一个新的染色体矩阵。

Encoding
OldChrom
FieldDR
Pm:      默认是 1/染色体长度
Sigma3:  3倍高斯变异的标准差，默认是 0.5*(ub-lb)/3。
Middle:  变异的中心是否是搜索域的中央，默认False，以变异前的值作为中心。
FixType
Parallel
"""


class Mutgau:
    def __init__(self, Encoding, FieldDR, Pm=None, Sigma3=None, Middle=None, FixType=None, Parallel=False):
        self.Encoding = Encoding
        self.FieldDR = FieldDR
        self.Pm = Pm
        self.Sigma3 = Sigma3
        self.Middle = Middle
        self.FixType = FixType
        self.Parallel = Parallel

    def do(self, Parent1):
        Offspring = ea.mutgau(self.Encoding, Parent1, self.FieldDR, self.Pm, self.Sigma3, self.Middle, self.FixType)
        return Offspring


class MutM2m:
    """
    MOEA/D-M2M
    """

    def __init__(self, FieldDR, maxgen) -> None:
        # self.D = None          # 决策变量维度
        self.FieldDR = FieldDR
        self.MaxGen = maxgen
        pass

    def do(self, OldChrom, r0, Parent1, currentGen: int):
        """
        M2m里的变异
        Parent1是需要变异的决策矩阵,shape=(1,D)
        r0是要变异的个体索引
        """
        # 变异
        # Parent1 = OldChrom[r0]
        N, D = Parent1.shape  # 决策变量维度
        rm = 0.25 * (2 * np.random.rand(N, D) - 1) * (1 - np.random.rand(N, D) ** (-(1 - currentGen / self.MaxGen) ** 0.7))
        Site = np.random.rand(N, D) < 1 / D
        # print(Site)
        Lower = np.tile(self.FieldDR[0], (N, 1))
        Upper = np.tile(self.FieldDR[1], (N, 1))
        OffDec = Parent1.copy()
        OffDec[Site] = OffDec[Site] + rm[Site] * (Upper[Site] - Lower[Site])

        # 边界处理
        temp1 = OffDec < Lower
        temp2 = OffDec > Upper
        rnd = np.random.rand(N, D)
        OffDec[temp1] = Lower[temp1] + 0.5 * rnd[temp1] * (OldChrom[r0:r0 + 1][temp1] - Lower[temp1])
        OffDec[temp2] = Upper[temp2] - 0.5 * rnd[temp2] * (Upper[temp2] - OldChrom[r0:r0 + 1][temp2])
        # 检查是否符合边界约束
        if np.any(OffDec > Upper) or np.any(OffDec < Lower):
            print("有解越界")
            print(OffDec)
        return OffDec


class ProcessBound:
    """
    空操作，仅处理边界信息
    """

    def do(self, OldChrom, FieldDR):
        lb = FieldDR[0]
        ub = FieldDR[1]
        # 不改变原来染色体矩阵
        OffChrom = OldChrom.copy()
        # 边界处理
        OffChrom = np.minimum(np.maximum(OffChrom, lb), ub)
        return OffChrom


if __name__ == '__main__':
    # 检查RecM2m
    p1 = np.array([1, 2, 3])
    p2 = np.array([2, 1, 4])
    randd = np.array([0.3])
    currentGen = 6
    MaxGen = 13
    rc = (2 * randd - 1) * (1 - randd ** (-(1 - currentGen / MaxGen) ** 0.7))
    OffDec = p1 + rc * (p1 - p2)
    print("交叉结果：")
    print(OffDec)
    D = 3
    randn = np.array([0.2, 0.5, 0.7])
    rm = 0.25 * (2 * randn - 1) * (1 - randn ** (-(1 - currentGen / MaxGen) ** 0.7))
    Site = randn < 1 / D
    Lower = np.array([1, 1, 1])
    Upper = np.array([3, 3, 3])
    # OffDec = OldChrom.copy()
    OffDec[Site] = OffDec[Site] + rm[Site] * (Upper[Site] - Lower[Site])
    print("变异结果")
    print(OffDec)
