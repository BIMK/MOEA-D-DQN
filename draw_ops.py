import time
from types import ModuleType
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

ops = np.load('./ops_/ops_moeaddqn_UF1_1.npy')
ops = np.r_[ops, ops[-2:-1, :]]
N, D = ops.shape
labels = ['OP1', 'OP2', 'OP3', 'OP4']
markers = ['o', '^', 's', 'p']
for i in range(4):
    plt.plot(ops[:, i], '.', label=labels[i], marker=markers[i])

plt.legend(loc='upper left', prop={'family': 'Times New Roman'})
xticks = list(map(str, np.arange(0, N + 1, 5)))
plt.xticks(range(0, N + 1, 5), xticks, fontsize=12)
plt.yticks(fontsize=12)
# 坐标轴名称
plt.xlabel('Generation (×10)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('Percentage of operators applied', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.savefig('C:/Users/lxp/Desktop/pic/' + "ops_uf1_" + str(int(time.time())) + '.pdf')
plt.show()
