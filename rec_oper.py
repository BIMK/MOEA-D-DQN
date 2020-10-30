import random
import numpy as np
import geatpy as ea
"""
交叉算子
------
SBX 模拟二进制交叉
DE  各种差分算子
moea/d-m2m 的交叉算子
"""

"""
模拟二进制交叉在geatpy中已有实现
class Recsbx:
    def __init__(XOVR=0.7, Half=False, n=20, Parallel=False):
        pass
    def do(OldChrom):
        pass
"""


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




