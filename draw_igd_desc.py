import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

immoea = './igd_desc/immoea_uf1.mat'
moead = './igd_desc/moead_uf1.mat'
moeaddqn = './igd_desc/moeaddqn_uf1.mat'
moeaddra = './igd_desc/moeaddra_uf1.mat'
moeaddyts = './igd_desc/moeaddyts_uf1.mat'
moeadfrrmab = './igd_desc/moeadfrrmab_uf1.mat'
moeadm2m = './igd_desc/moeadm2m_uf1.mat'

igds = [immoea, moead, moeaddqn, moeaddra, moeaddyts, moeadfrrmab, moeadm2m]
labels = ['IM-MOEA', 'MOEA/D', 'MOEA/D-DQN', 'MOEA/D-DRA', 'MOEA/D-DYTS', 'MOEA/D-FRRMAB', 'MOEA/D-M2M']
styles = ['o-', '^:', 's-', '-.x', '--o', ':*', ':D']

for i in range(len(igds)):
    igd = scipy.io.loadmat(igds[i])['igd_desc'][0]
    D = igd.shape[0]
    idx = np.linspace(0, D - 1, num=19, dtype='int32')
    igd = igd[idx]
    # igd = np.r_[igd, igd[-1]]
    plt.plot(igd, styles[i], label=labels[i])

# 坐标轴刻度
xticks = list(map(str, np.arange(1, 11, 1)))
plt.xticks(range(0, 20, 2), xticks)

# 坐标轴名称
plt.xlabel('Generation (×30)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('IGD', fontdict={'family': 'Times New Roman', 'fontsize': 14})

plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('C:/Users/lxp/Desktop/pic/' + 'igd_desc_' + str(int(time.time())) + '.pdf')
plt.show()
