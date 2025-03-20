import time

from initial_validation import config
from initial_validation.genetic import encoding, termination, decoding, genetic



def ga_new(parameters, init_method):

    # i = 1
    # for job in parameters["jobs"]:
    #     j = 1
    #     for op in job:
    #         j = j + 1
    #     i = i + 1

    # 设置当前时间
    t0 = time.time()

    # Initialize the Population
    #init_method = "heuristic"  # random/heuristic/mixed
    # 2.初始化种群并设置Gen，Gen为当前代;
    if init_method == "random":
        population = encoding.initializePopulation_random(parameters)
    elif init_method == "heuristic":
        # 基于启发式的，每次生成的种群是完全一样的
        population = encoding.initializePopulation_heuristic(parameters)
    elif init_method == "mixed":
        population = encoding.initializePopulation_mixed(parameters)
    else:
        population = []
    print("初始化种群：")
    print(population)

    # print(population)
    # 种群结构：[([OS1], [MS1]), ([OS2], [MS2]), ...]
    gen = 1

    # Evaluate the population
    # 3.根据目标评估种群中的每一个个体
    while not termination.shouldTerminate(population, gen):
        # Genetic Operators
        # 对种群进行遗传操作
        population = genetic.selection(population, parameters)
        population = genetic.crossover(population, parameters)
        population = genetic.mutation(population, parameters)

        gen = gen + 1

    sortedPop = sorted(population, key=lambda cpl: genetic.timeTaken(cpl, parameters))

    t1 = time.time()
    total_time = t1 - t0
    print("Finished in {0:.2f}s".format(total_time))

    # Termination Criteria Satisfied ?
    # 4.是否满足终止条件?
    gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1]))
    #print(gantt_data)

    # if config.latex_export:
    #     gantt.export_latex(gantt_data)
    # else:
    #     gantt.draw_chart(gantt_data)

    # 获取算法输出最优解makespan
    max_end_time = 0
    # 遍历每个机器的任务列表
    for machine, tasks in gantt_data.items():
        for task in tasks:
            # 获取任务的结束时间
            end_time = task[1]
            # 更新最大完工时间
            if end_time > max_end_time:
                max_end_time = end_time
    # 输出最大完工时间
    #print("最大完工时间:", max_end_time)
    return max_end_time