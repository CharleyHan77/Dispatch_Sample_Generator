# import random
#
# import numpy as np
# from deap import algorithms, creator, base, tools
# from scipy.stats import bernoulli
#
# random.seed(42) #确保可以复现结果
# # 描述问题
# # 我们要解决的是哪个多项式的最大值为题
# # 而且编码用list来表示，遗传算法中的每一个个体是[1,0,1,1,1,1,0,0,...]这样的list
# creator.create('FitnessMax', base.Fitness, weights=(1.0,)) # 单目标，最大值问题
# creator.create('Individual', list, fitness = creator.FitnessMax) # 编码继承list类
#
# # 个体编码
# GENE_LENGTH = 26 # 需要26位编码
# # creator上定义了目标问题和个体的表示方式list，然后再toolbox中进行注册。
# # toolbox定义编码方式二进制，在toolbox中注册个体
# toolbox = base.Toolbox()
# toolbox.register('binary', bernoulli.rvs, 0.5) #注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
# toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = GENE_LENGTH) #用tools.initRepeat生成长度为GENE_LENGTH的Individual
#
# # 评价函数，求极大值的函数
# def decode(individual):
#     num = int(''.join([str(_) for _ in individual]), 2) # 解码到10进制
#     x = -30 + num * 60 / (2**26 - 1) # 映射回-30，30区间
#     return x
#
# def eval(individual):
#     x = decode(individual)
#     return ((np.square(x) + x) * np.cos(2*x) + np.square(x) + x),
#
# # 生成初始族群
# # 之前在toolbox中注册了个体，现在重复个体，注册一个群体
# N_POP = 100 # 族群中的个体数量
# toolbox.register('population', tools.initRepeat, list, toolbox.individual)
# pop = toolbox.population(n = N_POP)
#
# # 在工具箱中注册遗传算法需要的工具
# # 之后的交叉，重组，选择，突变也要一一在toolbox中注册
# # 先注册评价函数，eval是之前定义的函数名字
# # 定义选择配种人群的方式，这里采用的是锦标赛选择
# # 然后重组采用的是均匀重组
# # 然后选择一个突变的方式
# toolbox.register('evaluate', eval)
# toolbox.register('select', tools.selTournament, tournsize = 2) # 注册Tournsize为2的锦标赛选择
# toolbox.register('mate', tools.cxUniform, indpb = 0.5) # 注意这里的indpb需要显示给出
# toolbox.register('mutate', tools.mutFlipBit, indpb = 0.5)
#
# # 注册计算过程中需要记录的数据
# stats = tools.Statistics(key=lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)
#
# # 调用DEAP内置的算法
# # 这里采用简单进化算法，所以用了eaSimple
# resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats = stats, verbose = False)
#
# # 输出计算过程
# logbook.header = 'gen', 'nevals',"avg", "std", 'min', "max"
# print(logbook)
