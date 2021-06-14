import numpy as np
from crossover import RecM2m
from mutation import MutM2m

# np.random.seed(1)
xovOper = RecM2m(100)

OldChrom = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.2, 0.5, 0.6, 0.1, 0.8]])
off = xovOper.do(OldChrom, 0, [1], 34)

print(off)

FieldDR = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])

mutOper = MutM2m(FieldDR, 100)
off = mutOper.do(OldChrom, 0, off, 34)
print(off)

# %%
import numpy as np
currentGen = 20
MaxGen = 200
rc = (2 * np.random.rand(1) - 1) * (1 - np.random.rand(1) ** (-(1 - currentGen / MaxGen) ** 0.7))
print(rc)
p1 = np.random.rand(1, 4)
p2 = np.random.rand(1, 4)
print(p1)
print(p2)
OffDec = p1 + rc * (p1 - p2)
print(OffDec)
# %%
