import random


# 1.选择操作（锦标赛选择）
def selection(population, fjsp, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        candidates = random.sample(population, tournament_size)
        best = min(candidates, key=lambda ind: fjsp.makespan(ind))
        selected.append(best)
    return selected

# 2.交叉操作（单点交叉）
def crossover(parent1, parent2):
    child1, child2 = [], []
    for p1_job, p2_job in zip(parent1, parent2):
        if random.random() < 0.5:  # 50% 的概率交叉
            child1.append(p1_job)
            child2.append(p2_job)
        else:
            child1.append(p2_job)
            child2.append(p1_job)
    return child1, child2

# 3.变异操作（随机交换两个操作）
def mutation(schedule, mutation_rate=0.1):
    for job in schedule:
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(job)), 2)
            job[idx1], job[idx2] = job[idx2], job[idx1]
    return schedule


def genetic_algorithm(fjsp, population, generations):
    convergence_curve = []  # 记录每代的最优 makespan
    for _ in range(generations):
        # 选择
        population = selection(population, fjsp)
        # 交叉
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        # 变异
        population = [mutation(ind) for ind in new_population]

        # 记录当前代的最优 makespan
        best_schedule = min(population, key=lambda ind: fjsp.makespan(ind))
        best_makespan = fjsp.makespan(best_schedule)
        convergence_curve.append(best_makespan)

    best_schedule = min(population, key=lambda ind: fjsp.makespan(ind))
    # 返回最优解
    return min(population, key=lambda ind: fjsp.makespan(ind))


