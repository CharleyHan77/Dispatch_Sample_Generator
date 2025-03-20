# import random
# from deap import base, creator, tools
# import numpy as np
#
#
#
# # Step 2: 初始化种群的个体生成函数
# def create_individual(init_individual):
#     return [list(ind) for ind in init_individual]  # 复制初始化种群
#
#
# # Step 3: 适应度函数 - 计算总完工时间 (makespan)
# def evaluate(individual):
#     # 计算每个工序在机器上的加工时间，并返回总完工时间
#     num_machines = len(individual[0])  # 获取机器数量
#     machine_times = [0] * num_machines  # 每台机器的当前时间
#
#     for job in individual:
#         for op in job:
#             machine_id, processing_time = op
#             machine_times[machine_id - 1] += processing_time  # 累积该机器的加工时间
#
#     makespan = max(machine_times)  # makespan 是所有机器的最大完成时间
#     return makespan,
#
#
# # Step 4: 遗传算法操作（选择、交叉、变异等）
#
# # 交叉操作：交换两个个体的某些工序
# def crossover(ind1, ind2):
#     size = len(ind1)
#     cxpoint = random.randint(1, size - 1)
#     ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
#     return ind1, ind2
#
#
# # 变异操作：随机选择一个工序，改变其机器或加工时间
# def mutate(individual):
#     job_idx = random.randint(0, len(individual) - 1)
#     op_idx = random.randint(0, len(individual[job_idx]) - 1)
#     machine_id, processing_time = individual[job_idx][op_idx]
#
#     # 随机改变机器编号或加工时间
#     new_machine_id = random.randint(1, len(individual[job_idx]))
#     new_processing_time = random.randint(1, 10)  # 假设加工时间在1到10之间
#     individual[job_idx][op_idx] = (new_machine_id, new_processing_time)
#     return individual,
#
#
# # Step 5: 注册遗传算法的操作
# toolbox = base.Toolbox()
#
# # 个体生成
# toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# # 选择操作
# toolbox.register("select", tools.selTournament, tournsize=3)
#
# # 交叉操作
# toolbox.register("mate", crossover)
#
# # 变异操作
# toolbox.register("mutate", mutate)
#
# # 适应度评估
# toolbox.register("evaluate", evaluate)
#
#
# # Step 6: 运行遗传算法
# def new_ga(init_population, generations):
#     # 初始化种群
#     population = toolbox.population(n=generations, init_individual=init_population)
#
#     # 运行遗传算法
#     for gen in range(50):  # 假设运行50代
#         print(f"Generation {gen}")
#
#         # 评估所有个体
#         fitnesses = list(map(toolbox.evaluate, population))
#         for ind, fit in zip(population, fitnesses):
#             ind.fitness.values = fit
#
#         # 选择：选择父代
#         offspring = toolbox.select(population, len(population))
#         offspring = list(map(toolbox.clone, offspring))
#
#         # 交叉：对父代进行交叉操作
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < 0.7:  # 70%的概率进行交叉
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values
#
#         # 变异：对部分个体进行变异
#         for mutant in offspring:
#             if random.random() < 0.2:  # 20%的概率进行变异
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values
#
#         # 评估所有个体
#         invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = list(map(toolbox.evaluate, invalid_individuals))
#         for ind, fit in zip(invalid_individuals, fitnesses):
#             ind.fitness.values = fit
#
#         # 替换旧一代
#         population[:] = offspring
#
#     # 输出最优解
#     best_individual = tools.selBest(population, 1)[0]
#     print(f"Best solution: {best_individual}, Fitness: {best_individual.fitness.values}")
#     return best_individual.fitness.values
#
#
# # if __name__ == "__main__":
# #     # 假设你已经有了初始化种群的结构
# #     init_population = [[[(1, 5), (5, 3), (3, 4), (6, 5), (3, 1), (4, 3)],
# #                         [(2, 6), (3, 1), (1, 2), (4, 6), (6, 5)],
# #                         [(2, 6), (3, 4), (1, 1), (3, 4), (5, 5)]]]
# #     main(init_population)
