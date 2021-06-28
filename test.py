import scipy.io
import numpy
metric = {
    'runtime': 4,
    'IGD': 1.12,
    'HV': 1.12
}
res = {'result': [10000, 123], 'metric': metric}
scipy.io.savemat('aos.mat', mdict=res)
