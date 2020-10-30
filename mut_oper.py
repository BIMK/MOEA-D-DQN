import random
import numpy as np
import geatpy as ea
"""
变异算子
mutpolyn 多项式变异
mutGas   高斯变异
moea/d-m2m的变异
"""

"""
多项式变异在geatpy中已有实现
class Mutpolyn:
    def __init__(Pm=None, DisI=20, FixType=1, Parallel=False):
        pass
    def do(Encoding, OldChrom, FieldDR):
        pass
"""

"""
高斯变异在geatpy中已有实现
class Mutgau:
    def __init__(Pm=None, Sigma3=False, Middle=False, FixType=1, Parallel=False):
        pass
    def do(Encoding, OldChrom, FieldDR):
        pass
"""


class MutM2m:
    """
    MOEA/D-M2M
    """
    
    def __init__(self, maxgen:int) -> None:
        # self.D = None          # 决策变量维度
        self.MaxGen = maxgen
    
    def do(self, Encoding, OldChrom, FieldDR, Parent1, currentGen: int):
        """
        M2m里的变异，边界处理时需要父代Parent1
        """
        # 变异
        D = len(Parent1)
        rm = 0.25 * (2 * np.random.rand(D) - 1) * (1 - np.random.rand(D) ** (-(1 - currentGen / self.MaxGen) ** 0.7))
        Site = np.random.rand(D) < 1 / D
        Lower = FieldDR[0]
        Upper = FieldDR[1]
        OffDec = OldChrom.copy()
        OffDec[Site] = OffDec[Site] + rm[Site] * (Upper[Site] - Lower[Site])
        
        # 边界处理
        temp1 = OffDec < Lower
        temp2 = OffDec > Upper
        rnd = np.random.rand(D)
        OffDec[temp1] = Lower[temp1] + 0.5 * rnd[temp1] * (Parent1[temp1] - Lower[temp1])
        OffDec[temp1] = Upper[temp2] - 0.5 * rnd[temp2] * (Upper[temp2] - Parent1[temp2])
        return OffDec

