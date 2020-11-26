# -*- coding: utf-8 -*-
import logging
import os
import sys
import time
import geatpy as ea  # import geatpy
import numpy as np

from moea_MOEAD_DRA_templet import moea_MOEAD_DRA_templet

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
# 配置日志信息
logging.basicConfig(
    handlers=[logging.FileHandler("./result/RL_UF1-7_gammaSW_0.log", encoding="utf-8", mode='w')],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S')
console = logging.StreamHandler()
# 定义一个Handler打印INFO及以上级别的日志到sys.stderr
console.setLevel(logging.INFO)
# 设置日志打印格式
formatter = logging.Formatter('%(levelname)-8s--> %(message)s')
console.setFormatter(formatter)
# 将定义好的console日志handler添加到root logger
logging.getLogger('').addHandler(console)
logging.info('Start----RL_UF1-7-----------------------------------')

def get_time():
    ans = time.strftime("%m-%d %H:%M:%S", time.localtime())
    return ans

if __name__ == '__main__':

    problems = ['UF1', 'UF2', 'UF3', 'UF4', 'UF5', 'UF6', 'UF7']
    # problems = ['UF8','UF9','UF10']
    # problems = ['DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7',]
    N = 35   # 独立运行N次，取中值
    results = list()
    for problemName in problems:
        """======================实例化问题对象========================="""
        logging.info('time: %s       Start ... %s' % (get_time(), problemName))
        fileName = problemName
        MyProblem = getattr(__import__('problem.'+problemName), problemName)  # 导入自定义问题类
        MyProblem = getattr(MyProblem, problemName)
        problem = MyProblem()       # 生成问题对象--DTLZ设置为3目标
        PF = problem.getReferObjV()  # 获取真实前沿，详见Problem.py中关于Problem类的定义
        """======================种群设置==============================="""
        Encoding = 'RI'             # 编码方式
        NIND = 600                  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)
        """======================算法参数设置=========================="""
        # myAlgorithm = moea_MOEAD_DE_templet(problem, population)
        MAXGEN = 500
        myAlgorithm = moea_MOEAD_DRA_templet(problem, population, MAXGEN)
        myAlgorithm.MAXGEN = MAXGEN    # 最大进化代数
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        igd = np.empty(N)
        hv = np.empty(N)
        for i in range(N):
            """==========================调用算法模板进行种群进化========================"""
            NDSet, pop, plt = myAlgorithm.run()  # 每次run都是独立的，会重新初始化模版的一些动态参数
            IGD = ea.indicator.IGD(pop.ObjV, PF)     # 计算IGD指标
            HV = ea.indicator.HV(NDSet.ObjV, PF)       # 计算HV指标
            logging.info('time: %s,  round %d/%d, IGD = %.7f, HV = %.7f' % (get_time(), i+1, N, IGD, HV))
            igd[i] = IGD
            hv[i] = HV
        # 去除5个最差的实验数据
        igd = np.sort(igd)[:-5]
        hv = np.sort(hv)[5:]
        res = ('median ----- IDG = %.7f, IGD.std = %.7f, HV = %.7f, HV.std = %.7f' %
                     (np.median(igd), np.std(igd), np.median(hv), np.std(hv)))
        results.append(res)
        logging.info(res)
        # logging.info('median ----- IDG = %.7f, IGD.std = %.7f, HV = %.7f, HV.std = %.7f' %
        #              (np.median(igd), np.std(igd), np.median(hv), np.std(hv)))
    for i in range(len(problems)):
        logging.info(problems[i], results[i])
    sys.exit(0)

"""
    NDSet,pop = myAlgorithm.run()   # 执行算法模板，得到帕累托最优解集NDSet
    # NDSet.save()                # 把结果保存到文件中
    # 输出
    # print(myAlgorithm.mutDE.name)
    print('用时：%s 秒'%(myAlgorithm.passTime))
    print('评价次数：%d 次'%(myAlgorithm.evalsNum))
    print('非支配个体数：%d 个'%(NDSet.sizes))
    # 计算指标
    PF = problem.getReferObjV() # 获取真实前沿，详见Problem.py中关于Problem类的定义
    if PF is not None and NDSet.sizes != 0:
        # GD = ea.indicator.GD(NDSet.ObjV, PF)       # 计算GD指标
        IGD = ea.indicator.IGD(NDSet.ObjV, PF)     # 计算IGD指标
        HV = ea.indicator.HV(NDSet.ObjV, PF)       # 计算HV指标
        # Spacing = ea.indicator.Spacing(NDSet.ObjV) # 计算Spacing指标
        # print('GD',GD)
        print('IGD',IGD)
        print('HV', HV)
        # print('Spacing', Spacing)
"""
"""=============================进化过程指标追踪分析============================"""
"""
    # if PF is not None:
    #     metricName = [['IGD'], ['HV']]
    #     [NDSet_trace, Metrics] = ea.indicator.moea_tracking(myAlgorithm.pop_trace, PF, metricName, problem.maxormins)
    #     # 绘制指标追踪分析图
    #     ea.trcplot(Metrics, labels = metricName, titles = metricName)
"""
