# %%
import numpy as np
from numpy.core.fromnumeric import size
a = np.array([[15.6882, 0.2671, 58.1048, 25.4084, 0.4643, 0.1671]])
print(a)
idx = np.argsort(a[0])
idx = 5 - np.argsort(idx)
print(idx)
proba = np.array([[70, 28, 10, 8, 5, 4]])
ac = proba[:, idx]
print(ac)
