"""
交叉算子池，交叉算子统一不处理边界信息
1. 模拟二进制交叉
2. 差分
3. MOEA/D-M2M里的交叉
"""
import numpy as np
import geatpy as ea


class RecRL:
    """
    用强化学习选择交叉算子
    """
    def __init__(self, problem, lambda_, dqn) -> None:
        self.name = "RecRL"
        self.problem = problem
        self.lambda_ = lambda_
        self.dqn = dqn
        self.recOpers = [ea.Recsbx(XOVR=0.7, Half=True, n=20), DE_rand_1(), DE_rand_2(),
                            DE_current_to_rand_1(), DE_current_to_rand_2()]
        self.n = len(self.recOpers)
        self.CountOpers = np.zeros(self.n)  # 统计不同算子的选择频率
    
    def do(self, OldChrom, r0, neighbourVector):
        """
        OldChrom: 变异前的原始矩阵
        """


"""
模拟二进制交叉 Simulated Binary Crossover  geatpy有实现
class recsbx(OldChrom, XOVR, Half, n, Parallel)

OldChrom: 待交叉种群染色体矩阵
XOVR:     交叉概率，默认0.7
Half:     默认False，当为True时，每对交叉结果只选择第一个保留。
n   :     分布指数，默认20，越大交叉结果越接近双亲
Parallel: False
"""

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
        x = OldChrom[r0:r0+1,:]  # 保持x是二维
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
        x = OldChrom[r0:r0+1,:]
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
        x = OldChrom[r0:r0+1,:]
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
        x = OldChrom[r0:r0+1,:]
        xr1 = OldChrom[r1]
        xr2 = OldChrom[r2]
        xr3 = OldChrom[r3]
        xr4 = OldChrom[r4]
        xr5 = OldChrom[r5]
        v = x + self.K * (x - xr1) + self.F * (xr2 - xr3 + xr4 - xr5)
        return v


class RecM2m:
    """
    MOEA/D-M2M里的交叉
    """
    def __init__(self) -> None:
        pass
    def do(self, OldChrom, neighbourVector, currentGen:int, MaxGen:int):
        r1 = neighbourVector[0]
        r2 = neighbourVector[1]
        p1 = OldChrom[r1]
        p2 = OldChrom[r2]
        # D  = len(p1)   #  决策变量维度
        rc = (2*np.random.rand(1)-1) * (1-np.random.rand(1) ** (-(1-currentGen/MaxGen)**0.7))
        OffDec = p1 + rc*(p1-p2)
        return OffDec


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

    Chrom = np.array([[1,2,3],[4,5,6]])
    rec = ea.Recsbx(Half=True)
    off = rec.do(Chrom)
    print(off)

    
