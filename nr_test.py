import time
from types import ModuleType
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


data = [0.0409349, 0.0174856, 0.0127900, 0.0100230, 0.0100284, 0.0067394, 0.0061007, 0.0046483]  # zdt1 igd
data = [0.6500695, 0.6942234, 0.7025797, 0.7085415, 0.7077856, 0.7138125, 0.7153235, 0.7180088]  # zdt hv
data = [0.0020321, 0.0012992, 0.0012278, 0.0012377, 0.0012482, 0.0012613, 0.0012606, 0.0012938]  # uf1 igd
data = [0.7212440, 0.7224687, 0.7226212, 0.7224939, 0.7226031, 0.7224985, 0.7223467, 0.7223118]  # uf1 hv

xticks = [5, 10, 15, 20]

plt.xticks(range(4), xticks)
plt.plot(data)
plt.plot(data, 'o')
plt.xlabel('neighborhood size', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('HV', fontdict={'family': 'Times New Roman', 'fontsize': 14})
# plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('C:/Users/lxp/Desktop/pic/' + 'nr_hv_zdt1_' + str(int(time.time())) + '.pdf')
plt.show()


'''
# 坐标轴刻度
# xticks = list(map(str, np.arange(0, 51, 5)))
# plt.xticks(range(0, 51, 5), xticks)
xticks = list(map(str, np.arange(1, 11)))
plt.xticks(range(0, 10), xticks)

# 坐标轴名称
plt.xlabel('Generation (×10)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
plt.ylabel('IGD', fontdict={'family': 'Times New Roman', 'fontsize': 14})

plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('C:/Users/lxp/Desktop/pic/' + 'igd_desc_zdt1' + str(int(time.time())) + '.pdf')
plt.show()

'''
