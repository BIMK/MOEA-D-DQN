import numpy as np
import optproblems.wfg
from optproblems import Individual
import geatpy as ea
from matplotlib import pyplot as plt


class WFG9(ea.Problem):
    def __init__(self, M=3) -> None:
        name = 'WFG9'
        maxormins = [1] * M
        Dim = M + 9
        varTypes = [0] * Dim
        lb = [0] * Dim
        ub = list(range(2, 2 * Dim + 1, 2))  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.n_obj = M
        self.n_var = Dim
        self.n_position_params = 2
        self.n_constr = 0
        self.func = optproblems.wfg.WFG9(self.n_obj, self.n_var, self.n_position_params)
        self.xl = self.func.min_bounds
        self.xu = self.func.max_bounds

    def aimFunc(self, pop):
        # x = pop.Phen
        x = pop
        solutions = [Individual(s) for s in x]
        self.func.batch_evaluate(solutions)
        res = np.array([s.objective_values for s in solutions])
        # pop.ObjV = res
        return res

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = 10000  # 设置所要生成的全局最优解的个数
        Point, num = ea.crtup(self.M, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point**2, 1)]).T), (1, self.M))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.M + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    wfg9 = WFG9()
    # chrom = np.random.random((3, 12))
    chrom = np.array([[0.15978844,
                       0.12375592,
                       0.2241123,
                       0.68292049,
                       0.81240837,
                       0.25309515,
                       0.43564012,
                       0.69708085,
                       0.01889147,
                       0.95138774,
                       0.27971926,
                       0.11375546],
                      [0.39005371,
                       0.52661483,
                       0.7541614,
                       0.38095297,
                       0.57416411,
                       0.03622639,
                       0.62139772,
                       0.82867097,
                       0.965468,
                       0.93487292,
                       0.12995088,
                       0.62697274],
                      [0.56406656,
                       0.19631747,
                       0.90129598,
                       0.24967638,
                       0.13175409,
                       0.20839596,
                       0.31978324,
                       0.3222819,
                       0.74744572,
                       0.44346628,
                       0.72374306,
                       0.90311056]])
    res = wfg9.aimFunc(chrom)
    print(res)
    # ref = wfg7.calReferObjV()
    # print(ref)
    # plt.plot(ref[:, 0], ref[:, 1], 'ob')
    # plt.show()
# %%
