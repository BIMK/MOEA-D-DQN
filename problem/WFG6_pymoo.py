import numpy as np
import geatpy as ea
from pymoo.model.problem import Problem
from pymoo.problems.many import generic_sphere
from pymoo.util.misc import powerset
from WFG1_pymoo import _transformation_shift_linear, _reduction_non_sep, _shape_concave


class WFG(Problem):

    def __init__(self, n_obj=3, **kwargs):
        n_var = n_obj + 9
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=0,
                         xu=2 * np.arange(1, n_var + 1),
                         type_var=np.double,
                         **kwargs)

        self.S = np.arange(2, 2 * self.n_obj + 1, 2)
        self.A = np.ones(self.n_obj - 1)
        self.k = n_obj - 1
        self.l = n_var - self.k
        self.validate(self.l, self.k, self.n_obj)

    def validate(self, l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        # if k < 4:
            # raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    def _post(self, t, a):
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x, s, h):
        return x[:, -1][:, None] + s * np.column_stack(h)

    def _rand_optimal_position(self, n):
        return np.random.random((n, self.k))

    def _positional_to_optimal(self, K):
        suffix = np.full((len(K), self.l), 0.35)
        X = np.column_stack([K, suffix])
        return X * self.xu

    def _calc_pareto_set(self, n_pareto_points=500, *args, **kwargs):
        ps = np.ones((2 ** self.k, self.k))
        for i, s in enumerate(powerset(np.arange(self.k))):
            ps[i, s] = 0

        rnd = self._rand_optimal_position(n_pareto_points - len(ps))
        ps = np.row_stack([ps, rnd])
        ps = self._positional_to_optimal(ps)
        return ps

    def _calc_pareto_front(self, *args, **kwargs):
        # ps = self.pareto_set(n_pareto_points=n_pareto_points)
        # return self.evaluate(ps, return_values_of=["F"])
        return None


class WFG6(WFG, ea.Problem):
    def __init__(self, n_obj=3):
        super().__init__()
        self.name = 'WFG6'
        self.M = n_obj
        self.maxormins = [1] * self.M
        # self.Dim = self.M+9
        self.Dim = self.n_var
        self.varTypes = [0] * self.Dim
        lb = [0] * self.Dim  # 决策变量下界
        ub = list(range(2, 2 * self.Dim + 1, 2))  # 决策变量上界
        self.ranges = np.array([lb, ub])
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.borders = np.array([lbin, ubin])

    def t1(self, x, n, k):
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @ staticmethod
    def t2(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def aimFunc(self, pop):
        x = pop.Phen
        y = x / self.xu
        y = self.t1(y, self.n_var, self.k)
        y = WFG6.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        pop.ObjV = self._calculate(y, self.S, h)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs) * self.S


class Population:
    def __init__(self) -> None:
        self.Phen = None
        self.ObjV = None


if __name__ == '__main__':
    wfg6 = WFG6()
    pop = Population()
    pop.Phen = np.random.random((3, 12))
    print(pop.Phen)
    wfg6.aimFunc(pop)
    print(pop.ObjV)
