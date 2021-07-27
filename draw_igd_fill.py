import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


x_dqn = np.arange(0, 499)
x_rand = np.arange(0, 499)
igd_dqn = np.zeros((20, 519))
igd_rand = np.zeros((20, 499))
for i in range(20):
    igd_dqn[i] = np.load('./igd_desc_rand/moeaddqn_1.0_UF1_' + str(i + 1) + '.npy')

for i in range(20):
    igd_rand[i] = scipy.io.loadmat('./igd_desc_rand/moeaddqnrand_uf1_ (' + str(i + 1) + ').mat')['igd_desc'][0]

igd_dqn = igd_dqn[:, :499]
mean_dqn = np.mean(igd_dqn, axis=0)
max_dqn = np.max(igd_dqn, axis=0)
min_dqn = np.min(igd_dqn, axis=0)

mean_rand = np.mean(igd_rand, axis=0)
max_rand = np.max(igd_rand, axis=0)
min_rand = np.min(igd_rand, axis=0)
plt.fill_between(x_dqn, min_dqn, max_dqn, alpha=0.2)
plt.fill_between(x_rand, min_rand, max_rand, alpha=0.2)
plt.plot(x_dqn, mean_dqn, label='DQN')
plt.plot(x_rand, mean_rand, label='random')
plt.legend()
plt.show()


# # load30次igd的结果
# N = 119
# igd = np.zeros((30, 119))
# x = np.arange(0, 119)
# dqn_igd = np.zeros_like(igd)
# freq = ['0.01', '0.2', '0.4', '0.6', '0.8', '1.0']
# freq_igd = np.zeros((6, 119))
# for k in range(6):
#     for i in range(30):
#         igd[i] = np.load('./igd_desc/moeaddqn_' + freq[k] + '_ZDT1_' + str(i + 1) + '.npy')
#         # dqn_igd[i] = np.load('./igd_desc/moeaddqn_' + str(i + 1) + '.npy')
#     freq_igd[k] = np.average(igd, axis=0)
#     plt.plot(x, freq_igd[k], label=freq[k])

# plt.legend()
# plt.show()

# max_igd = np.max(igd, axis=0)
# min_igd = np.min(igd, axis=0)
# mean_igd = np.mean(igd, axis=0)
# max_dqn_igd = np.max(dqn_igd, axis=0)
# min_dqn_igd = np.min(dqn_igd, axis=0)
# mean_dqn_igd = np.mean(dqn_igd, axis=0)

# plt.fill_between(x, min_igd, max_igd, alpha=0.2)
# plt.plot(x, mean_igd)
# plt.fill_between(x, min_dqn_igd, max_dqn_igd, alpha=0.2)
# plt.plot(x, mean_dqn_igd)
# plt.show()

""" for i in range(len(igds)):
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
plt.show() """
