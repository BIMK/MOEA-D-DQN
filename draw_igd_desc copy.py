import time
from types import ModuleType
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import math

""" immoea = './igd_desc/immoea'
moead = './igd_desc/moead'
moeaddqn = './igd_desc/moeaddqn'
moeaddra = './igd_desc/moeaddra'
moeaddyts = './igd_desc/moeaddyts'
moeadfrrmab = './igd_desc/moeadfrrmab'
# moeadm2m = './igd_desc/moeadm2m'
smea = './igd_desc/smea'
 """
moead = [0.067894, 0.014047, 0.014225, 0.036068, 0.03534, 0.24199, 0.093053, 0.087496, 0.17637, 0.074132]
moeaddra = [0.085451, 0.037158, 0.10573, 0.32283, 0.30633, 0.0067398, 0.0012058, 0.0023819, 0.0044942, 0.0011455]
immoea = [0.069842, 0.068311, 0.061863, 0.19492, 0.2003, 0.066506, 0.061211, 0.042495, 0.08302, 0.064022]
smea = [0.62046, 0.73509, 0.65989, 1.3843, 1.1777, 0.030426, 0.037335, 0.042544, 0.064587, 0.094069]
moeadfrrmab = [0.053026, 0.045727, 0.16183, 0.2206, 0.21064, 0.002062, 0.0015098, 0.001981, 0.0018082, 0.0022769]
moeaddyts = [0.037973, 0.065132, 0.44822, 0.099374, 0.24106, 0.001327, 0.001691, 0.0014136, 0.0031483, 0.0012018]
moeaddqn = [0.006452, 0.0055174, 0.004777, 0.0060235, 0.0052054, 0.0013137, 0.0012865, 0.0010346, 0.001289, 0.001682]


igds = [moead, moeaddra, immoea, smea, moeadfrrmab, moeaddyts, moeaddqn]
labels = ['MOEA/D', 'MOEA/D-DRA', 'IM-MOEA', 'SMEA', 'MOEA/D-FRRMAB', 'MOEA/D-DYTS', 'MOEA/D-DQN']
styles = ['s-', '^:', ':*', '-.x', '--o', ':D', 'o-']

for i in range(len(igds)):
    igd = igds[i][5:]
    # igd = math.log(igd)
    igd = [math.log10(x) for x in igd]
    plt.plot(igd, styles[i], label=labels[i])
# plt.legend()
# plt.show()

# exit()

# 坐标轴刻度
# xticks = ['ZDT1_30', 'ZDT1_50', 'ZDT1_80', 'ZDT1_100', 'ZDT1_150']
# xticks = ['UF1_30', 'UF1_50', 'UF1_80', 'UF1_100', 'UF1_150']
xticks = ['30', '50', '80', '100', '150']
plt.xticks(range(5), xticks)

# 坐标轴名称
plt.xlabel('Number of decision variables', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('IGD (logarithmic)', fontdict={'family': 'Times New Roman', 'fontsize': 14})

plt.legend(loc='upper left', prop={'family': 'Times New Roman'})
plt.savefig('C:/Users/lxp/Desktop/pic/' + 'igd_uf1_30-150' + str(int(time.time())) + '.pdf')
plt.show()
