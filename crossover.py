"""
交叉算子池，交叉算子统一不处理边界信息
1. 模拟二进制交叉
2. 差分
3. MOEA/D-M2M里的交叉
"""
import numpy as np
import geatpy as ea
from Nature_DQN import DQN


class RecRL:
    """
    用强化学习选择交叉算子
    """
    
    def __init__(self, problem, lambda_, maxgen, NIND) -> None:
        self.name = "RecRL"
        self.problem = problem
        self.lambda_ = lambda_
        self.maxgen = maxgen
        self.recOpers = [Recsbx(XOVR=0.7, Half=True, n=20), RecM2m(self.maxgen), DE_rand_1(), DE_rand_2(),
                         DE_current_to_rand_1(), DE_current_to_rand_2()]
        self.n = len(self.recOpers)
        self.dqn = DQN(problem.Dim, self.n)
        self.SW = np.zeros((2, NIND // 2))
        self.a = 0
        self.state = None
        self.state_ = None
        self.countOpers = np.zeros(self.n)  # 统计不同算子的选择频率
    
    def do(self, OldChrom, r0, neighbourVector, currentGen):
        """
        OldChrom: 变异前的原始矩阵
        return:  返回新种群的染色体矩阵
        """
        self.state = OldChrom[r0]
        self.a = self.dqn.choose_action(self.state)
        self.countOpers[self.a]+=1
        if self.a == 0:  # 使用模拟二进制交叉
            offChrom = self.recOpers[0].do(OldChrom, r0, neighbourVector)
        elif self.a == 1:  # 使用M2m里的交叉
            offChrom = self.recOpers[1].do(OldChrom, r0, neighbourVector, currentGen)
        else:  # 使用差分算子
            offChrom = self.recOpers[self.a].do(OldChrom, r0, neighbourVector)
        self.state_ = offChrom[0]
        return offChrom
    
    def learn(self, r):
        """
        更新DQN
        :param r: 子代相对于父代适应度的提高率
        子代和父代都是由上一步do得到的
        """
        self.SW = np.concatenate((self.SW[:, 1:], np.array([[self.a], [r]])), axis=1)

        # r = np.empty(self.n)
        # for i in range(n):
        #     r[i] = np.sum(self.SW[1, self.SW[0, :] == i])
            # self.DQN.store_transition(state,i,r[i],state_)
        r = np.sum(self.SW[1, self.SW[0,:] == self.a])
        self.dqn.store_transition(self.state, self.a, r, self.state_)
        # 学习,更新DQN
        if self.dqn.memory_counter > 100:
            self.dqn.learn()



"""
模拟二进制交叉 Simulated Binary Crossover  geatpy有实现
class Recsbx:
    def __init__(XOVR=0.7, Half=False, n=20, Parallel=False):
        pass
    def do(OldChrom):
        pass

OldChrom: 待交叉种群染色体矩阵
XOVR:     交叉概率，默认0.7
Half:     默认False，当为True时，每对交叉结果只选择第一个保留。
n   :     分布指数，默认20，越大交叉结果越接近双亲
Parallel: False
"""


class Recsbx:
    def __init__(self, XOVR=0.7, Half=True, n=20, Parallel=False):
        self.name = "Recsbx"
        self.XOVR = XOVR
        self.Half = Half
        self.n = n
        self.Parallel = Parallel
    
    def do(self, OldChrom, r0, neighbourVector):
        # pop[r0]作为父代,还需要另外一个父代才能做交叉,所以需要从neighbourVector里随机选一个
        r1 = np.random.choice(neighbourVector)
        p1 = np.vstack((OldChrom[r0], OldChrom[r1]))
        off = ea.recsbx(p1, self.XOVR, self.Half, self.n, self.Parallel)
        return off


class RecM2m:
    """
    MOEA/D-M2M里的交叉
    """
    
    def __init__(self, maxgen: int) -> None:
        self.name = 'RecM2m'
        self.MaxGen = maxgen
    
    def do(self, OldChrom, r0, neighbourVector, currentGen: int):
        # r1 = neighbourVector[0]
        # r2 = neighbourVector[1]
        r1, r2 = np.random.choice(neighbourVector, 2, replace=False)  # 不放回的抽取2个
        p1 = OldChrom[r0]
        # p1 = OldChrom[r1]
        p2 = OldChrom[r2]
        # D  = len(p1)   #  决策变量维度
        rc = (2 * np.random.rand(1) - 1) * (1 - np.random.rand(1) ** (-(1 - currentGen / self.MaxGen) ** 0.7))
        OffDec = p1 + rc * (p1 - p2)
        OffDec = OffDec[np.newaxis, :]
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