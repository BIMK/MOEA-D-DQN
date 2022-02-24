import time
from types import ModuleType
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

immoea = './igd_desc/immoea'
moead = './igd_desc/moead'
moeaddqn = './igd_desc/moeaddqn'
moeaddra = './igd_desc/moeaddra'
moeaddyts = './igd_desc/moeaddyts'
moeadfrrmab = './igd_desc/moeadfrrmab'
# moeadm2m = './igd_desc/moeadm2m'
smea = './igd_desc/smea'

igds = [moead, moeaddra, immoea, smea, moeadfrrmab, moeaddyts, moeaddqn]
labels = ['MOEA/D', 'MOEA/D-DRA', 'IM-MOEA', 'SMEA', 'MOEA/D-FRRMAB', 'MOEA/D-DYTS', 'MOEA/D-DQN']
styles = ['s-', '^:', ':*', '-.x', '--o', ':D', 'o-']

for i in range(len(igds)):
    igd = scipy.io.loadmat(igds[i] + '_zdt1.mat')['igd_desc'][0]
    igd = np.r_[igd, igd[-1]]
    D = igd.shape[0]
    idx = np.linspace(0, D - 1, num=10, dtype='int32')
    igd = igd[idx]
    plt.plot(igd, styles[i], label=labels[i], markevery=0.001)

# plt.show()
# exit()

# 坐标轴刻度
xticks = list(map(str, np.arange(1, 11, 1)))
plt.xticks(range(0, 10, 1), xticks)
# xticks = list(map(str, np.arange(1, 5, 30)))
# plt.xticks(range(0, 30), xticks)

# 坐标轴名称
plt.xlabel('Generation (×10)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('IGD', fontdict={'family': 'Times New Roman', 'fontsize': 14})

plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('C:/Users/lxp/Desktop/pic/' + 'igd_desc_zdt1' + str(int(time.time())) + '.pdf')
plt.show()
