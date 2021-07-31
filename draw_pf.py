import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from problem.ZDT1 import ZDT1
from problem.UF1 import UF1

problem = ZDT1()
# problem = UF1()
PF = problem.calReferObjV()

immoea = './pf/immoea'
moead = './pf/moead'
moeaddqn = './pf/moeaddqn'
moeaddra = './pf/moeaddra'
moeaddyts = './pf/moeaddyts'
moeadfrrmab = './pf/moeadfrrmab'
# moeadm2m = './pf/moeadm2m'
smea = './pf/smea'

igds = [moead, moeaddra, immoea, smea, moeadfrrmab, moeaddyts, moeaddqn]
# labels = [r'IM-MOEA', r'MOEA/D', r'MOEA/D-DQN', r'MOEA/D-DRA', r'MOEA/D-DYTS', r'MOEA/D-FRRMAB', r'MOEA/D-M2M']
labels = [r'MOEA/D', r'MOEA/D-DRA', r'IM-MOEA', r'SMEA', r'MOEA/D-FRRMAB', r'MOEA/D-DYTS', r'MOEA/D-DQN']
styles = ['o-', '^:', 's-', '-.x', '--o', ':*', ':D']

for i in range(len(igds)):
    pf = scipy.io.loadmat(igds[i] + '_uf1.mat')['objs']
    x = pf[:, 0]
    y = pf[:, 1]
    plt.plot(x, y, 'o', label=labels[i], markerfacecolor='none')
    plt.plot(PF[:, 0], PF[:, 1], label='PF')
    plt.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': '20'})
    # 坐标轴名称
    plt.xlabel('$f_1$', size=24)
    plt.ylabel('$f_2$', size=24)
    # plt.subplots_adjust(bottom=2)
    plt.tight_layout(pad=2.2)
    plt.savefig('C:/Users/lxp/Desktop/pic/' + igds[i].split('/')[-1] + '_pf_zdt1_' + str(int(time.time())) + '.pdf')
    plt.show()

# # 坐标轴刻度
# xticks = list(map(str, np.arange(1, 11, 1)))
# plt.xticks(range(0, 20, 2), xticks)

# # 坐标轴名称
# plt.xlabel('Generation (×30)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
# plt.ylabel('IGD', fontdict={'family': 'Times New Roman', 'fontsize': 14})

# plt.legend(prop={'family': 'Times New Roman'})
# plt.savefig('C:/Users/lxp/Desktop/pic/' + 'igd_desc_' + str(int(time.time())) + '.pdf')
# plt.show()
