# %%
# from pymop.problem import Problem
# import autograd.numpy as anp
import optproblems.wfg
import numpy as np
from optproblems import Individual
from matplotlib import pyplot as plt


class WFG():
    def __init__(self, n_obj=3, n_var=12, n_position_params=2):
        # super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0)
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = 0
        self.n_position_params = n_position_params

    def _evaluate(self, x, out=None, *args, **kwargs):
        solutions = [Individual(s) for s in x]
        self.func.batch_evaluate(solutions)
        res = np.array([s.objective_values for s in solutions])
        return res


class WFG6(WFG):
    def __init__(self, n_obj=3, n_var=12, n_position_params=2):
        super().__init__(n_obj=n_obj, n_var=n_var, n_position_params=n_position_params)
        self.func = optproblems.wfg.WFG6(n_obj, n_var, n_position_params)
        self.xl = self.func.min_bounds
        self.xu = self.func.max_bounds

    def calReferObjV(self, N):
        solutions = self.func.get_optimal_solutions(N)
        # solutions = [s.objective_values for s in solutions]
        self.func.batch_evaluate(solutions)
        res = np.array([s.objective_values for s in solutions])
        return res


# %%
if __name__ == '__main__':
    np.set_printoptions(precision=4)
    wfg6 = WFG6()
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
    res = wfg6._evaluate(chrom)
# %%
    print(res)
# %%
    # opt = wfg6.calReferObjV(100)
    # print(opt[:, 0].shape)
# %%
    # plt.plot(opt[:, 0], opt[:, 1], "ob")
    # plt.show()
