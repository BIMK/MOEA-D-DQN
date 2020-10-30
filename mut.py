"""
变异算子池，包括
1. 多项式变异
2. 高斯变异
3. MOEA/D-M2M的变异

应当先交叉再变异，因为变异算子会检查边界值

为了和geatpy库兼容，传入的仍旧是一个种群，只是种群里的个体只有一个。
"""
import numpy as np


class mutRL:
    def __init__(self) -> None:
        pass
    
    def do(self):
        pass


"""
多项式变异，geatpy已经提供
class mutpolyn(Encoding, OldChrom, FieldDR, Pm=20, FixType=4, Parallel=False)

Encoding: 染色体编码方式-RI
OldChrom: 种群染色体矩阵
FieldDR:  指明每个变量的上下界
Pm:       变异算子所发生作用的最小片段发生变异的概率
FixType:  用某种方式对超出范围的染色体进行修复

"""

"""
高斯变异，geatpy已经提供
class mutgau(Encoding, OldChrom, FieldDR, Pm, Sigma3, Middle, FixType, Parallel=False)

Encoding
OldChrom
FieldDR
Pm:      默认是 1/染色体长度
Sigma3:  3倍高斯变异的标准差，默认是 0.5*(ub-lb)/3。
Middle:  变异的中心是否是搜索域的中央，默认False，以变异前的值作为中心。
FixType
Parallel
"""

class MutM2m:
    """
    MOEA/D-M2M
    """
    def __init__(self) -> None:
        # self.D = None          # 决策变量维度
        # self.MaxGen = None
        pass
    def do(self, Encoding, OldChrom, FieldDR, Parent1, currentGen:int, MaxGen:int):
        """
        M2m里的变异，边界处理时需要父代Parent1
        """
        # 变异
        D = len(Parent1)
        rm     = 0.25*(2*np.random.rand(D)-1)*(1-np.random.rand(D)**(-(1-currentGen/MaxGen)**0.7))
        Site   = np.random.rand(D) < 1/D
        Lower  = FieldDR[0]
        Upper  = FieldDR[1]
        OffDec = OldChrom.copy()
        OffDec[Site] = OffDec[Site] + rm[Site]*(Upper[Site] - Lower[Site])
                
        # 边界处理
        temp1 = OffDec < Lower
        temp2 = OffDec > Upper
        rnd   = np.random.rand(D)
        OffDec[temp1] = Lower[temp1]+0.5*rnd[temp1]*(Parent1[temp1] - Lower[temp1])
        OffDec[temp1] = Upper[temp2]-0.5*rnd[temp2]*(Upper[temp2] - Parent1[temp2])
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
    p1 = np.array([1,2,3])
    p2 = np.array([2,1,4])
    randd = np.array([0.3])
    currentGen = 6
    MaxGen = 13
    rc = (2*randd-1) * (1-randd ** (-(1-currentGen/MaxGen)**0.7))
    OffDec = p1 + rc*(p1-p2)
    print("交叉结果：")
    print(OffDec)
    D = 3
    randn = np.array([0.2,0.5,0.7])
    rm     = 0.25*(2*randn-1)*(1-randn**(-(1-currentGen/MaxGen)**0.7))
    Site   = randn < 1/D
    Lower  = np.array([1,1,1])
    Upper  = np.array([3,3,3])
    # OffDec = OldChrom.copy()
    OffDec[Site] = OffDec[Site] + rm[Site]*(Upper[Site] - Lower[Site])
    print("变异结果")
    print(OffDec)





