import numpy as np
from crossover import RecM2m
from mutation import MutM2m

# np.random.seed(1)
xovOper = RecM2m(100)

OldChrom = np.array([[0.1,0.2,0.3,0.4,0.5,0.6],[0.3,0.2,0.5,0.6,0.1,0.8]])
off = xovOper.do(OldChrom, 0, [1], 34)

print(off)

FieldDR = np.array([[0,0,0,0,0,0],[1,1,1,1,1,1]])

mutOper = MutM2m(FieldDR, 100)
off = mutOper.do(OldChrom,0, off, 34)
print(off)




