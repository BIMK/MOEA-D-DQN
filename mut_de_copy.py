"""
可以接受一个种群传入的差分算子
"""
import numpy as np
from Nature_DQN import DQN

N_ACTIONS = 4   # 动作数量，等于候选算子数量，用于动作选择空间
N_STATES = 30  # 状态维度，等于决策变量维度，用于构造DQN的网络结构
MEMORY_CAPACITY = 100  # 学习MEMORY_CAPCITY个transaction后开始决策


class DE_rand_1():
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2 );
    """

    def __init__(self, F=0.5, CR=0.7, DN=1, Loop=False):
        self.name = "DE_rand_1"
        self.F = F  # 差分变异缩放因子
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 表示是否采用循环的方式处理超出边界的变异结果
        self.CR = CR   # 交叉概率

    def do(self, Encoding, OldChrom, FieldDR, *args):  # 执行变异
        """
        Encoding：编码方式
        OldChrom: 原始种群基因型
        """
        if len(args) != 1:
            raise RuntimeError('error in Mutde: Parameter error. (传入参数错误。)')
        r0_or_Xr0 = args[0]
        # return mutde(Encoding, OldChrom, FieldDR, r0_or_Xr0, self.F, self.DN, self.Loop)
        PopSize, ChromLength = OldChrom.shape
        NewChrom = np.zeros_like(OldChrom)
        # vi = xi + F × (xr1 − xr2 );
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        per = np.random.permutation(PopSize)
        for i in range(PopSize):
            # 变异
            x = OldChrom[r0_or_Xr0[i]]
            v = x+self.F * (OldChrom[per[i]] - OldChrom[per[(i+1) % PopSize]])
            # 交叉 伯努利分布交叉
            rand = np.random.binomial(1, self.CR, ChromLength)
            v[rand == 0] = x[rand == 0]
            # 边界处理
            lb = FieldDR[0]
            ub = FieldDR[1]
            v[v < lb] = lb[v < lb]
            v[v > ub] = ub[v > ub]
            NewChrom[i, :] = v
        return NewChrom

    def getHelp(self):  # 查看内核中的变异算子的API文档
        # help(mutde)
        print("比如")


class DE_rand_2():
    """
    差分进化，
        # vi = xi + F × (xr1 − xr2 + xr3 - xr4);
    """

    def __init__(self, F=0.5, CR=0.7, DN=1, Loop=False):
        self.name = "DE_rand_2"
        self.F = F  # 差分变异缩放因子
        self.DN = DN  # 表示有多少组差分向量
        self.Loop = Loop  # 表示是否采用循环的方式处理超出边界的变异结果
        self.CR = CR

    def do(self, Encoding, OldChrom, FieldDR, *args):  # 执行变异
        if len(args) != 1:
            raise RuntimeError('error in Mutde: Parameter error. (传入参数错误。)')
        r0_or_Xr0 = args[0]
        # return mutde(Encoding, OldChrom, FieldDR, r0_or_Xr0, self.F, self.DN, self.Loop)
        PopSize, ChromLength = OldChrom.shape
        NewChrom = np.zeros_like(OldChrom)
        # vi = xi + F × (xr1 − xr2 +xr3 -xr4);
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        per = np.random.permutation(PopSize)
        for i in range(PopSize):
            x = OldChrom[r0_or_Xr0[i]]
            v = x + self.F * (OldChrom[per[i]] - OldChrom[per[(i+1) % PopSize]] +
                              OldChrom[per[(i+2) % PopSize]] - OldChrom[per[(i+3) % PopSize]])
            # 交叉
            rand = np.random.binomial(1, self.CR, ChromLength)
            v[rand == 0] = x[rand == 0]
            # 边界处理
            lb = FieldDR[0]
            ub = FieldDR[1]
            v[v < lb] = lb[v < lb]
            v[v > ub] = ub[v > ub]
            NewChrom[i, :] = v
        return NewChrom

    def getHelp(self):  # 查看内核中的变异算子的API文档
        # help(mutde)
        print("比如")


class DE_current_to_rand_1():
    # vi=xi+K(xi-xr1)+F(xr2-xr3)
    def __init__(self, F=0.5, K=0.6):
        self.name = "DE_current_to_rand_1"
        self.F = F
        self.K = K

    def do(self, Encoding, OldChrom, FieldDR, xi):  # 执行变异
        """
        xi:基向量索引
        FieldDR：译码矩阵 [lb:变量下界;ub:变量上界;varTypes:0:连续1:离散]
        """
        PopSize, ChromLength = OldChrom.shape
        NewChrom = np.zeros_like(OldChrom)
        # vi=xi+K(xi-xr1)+F(xr2-xr3)
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        per = np.random.permutation(PopSize)
        for i in range(PopSize):
            x = OldChrom[xi[i]]
            xr1 = OldChrom[per[i]]
            xr2 = OldChrom[per[(i+1) % PopSize]]
            xr3 = OldChrom[per[(i+2) % PopSize]]
            v = x+self.K*(x-xr1)+self.F*(xr2-xr3)
            # 边界处理
            lb = FieldDR[0]
            ub = FieldDR[1]
            v[v < lb] = lb[v < lb]
            v[v > ub] = ub[v > ub]
            NewChrom[i, :] = v
        return NewChrom

    def getHelp(self):
        pass


class DE_current_to_rand_2():
    # v1=xi+K(xi-xr1)+F(xr2-xr3)+F(xr4-xr5)
    def __init__(self, F=0.5, K=0.6):
        self.name = "DE_current_to_rand_2"
        self.F = F
        self.K = K

    def do(self, Encoding, OldChrom, FieldDR, xi):  # 执行变异
        """
        xi:基向量索引
        FieldDR：译码矩阵 [lb:变量下界;ub:变量上界;varTypes:0:连续1:离散]
        """
        PopSize, ChromLength = OldChrom.shape
        NewChrom = np.zeros_like(OldChrom)
        # vi=xi+K(xi-xr1)+F(xr2-xr3)
        # xi基向量索引由传入的r0_or_Xr0确定，差分向量索引r1 r2按照随机方式确定,并尽可能保证互不相等
        per = np.random.permutation(PopSize)
        for i in range(PopSize):
            x = OldChrom[xi[i]]
            xr1 = OldChrom[per[i]]
            xr2 = OldChrom[per[(i+1) % PopSize]]
            xr3 = OldChrom[per[(i+2) % PopSize]]
            xr4 = OldChrom[per[(i+3) % PopSize]]
            xr5 = OldChrom[per[(i+4) % PopSize]]
            v = x+self.K*(x-xr1)+self.F*(xr2-xr3+xr4-xr5)
            # 边界处理
            lb = FieldDR[0]
            ub = FieldDR[1]
            v[v < lb] = lb[v < lb]
            v[v > ub] = ub[v > ub]
            NewChrom[i, :] = v
        return NewChrom
