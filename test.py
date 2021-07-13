# %%
import scipy.io
import numpy as np

a = np.array([[1, 2, 3]])
scipy.io.savemat('a.mat', {'igd_desc': a})
print(a.shape)

# %%
import numpy as np
a = np.random.randint(0, 100, 147)
print(a.shape)
idx = np.linspace(0, 146, num=99, dtype='int32')
print(idx)
print(a[idx])
