import time
from types import ModuleType
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# data = [0.0132389, 0.0088285, 0.0057234, 0.0048383]  # zdt igd
data = [0.7041312, 0.7101634, 0.7159579, 0.7178642]  # zdt hv
# data = [0.0020321, 0.0012821, 0.0011566, 0.0011537]  # uf1 igd
# data = [0.7210877, 0.7225783, 0.7227181, 0.7226936]  # uf1 hv

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
