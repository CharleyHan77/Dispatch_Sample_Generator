import numpy as np
import random
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from parse_fjs import parse
from validation import genetic_algorithm
from validation.FJSP_set import FJSP, parse_fjsp_data


# 1.初始种群1（生成规则：启发式-最短处理时间优先SPT）
def heuristic_schedule(fjsp):
    schedule = []
    machine_times = {machine: 0 for machine in fjsp.machines}
    for job in fjsp.jobs:
        job_schedule = []
        for operation in job:
            machines, times = operation
            # 选择可用时间最早的机器
            best_machine = min(machines, key=lambda m: machine_times[m])
            best_time = times[machines.index(best_machine)]
            job_schedule.append((best_machine, best_time))
            machine_times[best_machine] += best_time
        schedule.append(job_schedule)
    return schedule

# 2.初始种群2（生成规则：50% 部分随机 + 50% 部分启发式）
def mixed_schedule(fjsp):
    schedule = []
    machine_times = {machine: 0 for machine in fjsp.machines}
    for job in fjsp.jobs:
        job_schedule = []
        for operation in job:
            machines, times = operation
            if random.random() < 0.5:  # 50% 的概率使用启发式方法
                best_machine = min(machines, key=lambda m: machine_times[m])
                best_time = times[machines.index(best_machine)]
                job_schedule.append((best_machine, best_time))
            else:  # 50% 的概率随机选择机器
                idx = random.randint(0, len(machines) - 1)
                job_schedule.append((machines[idx], times[idx]))
            machine_times[job_schedule[-1][0]] += job_schedule[-1][1]
        schedule.append(job_schedule)
    return schedule

# 3.初始种群3（生成规则：完全随机）
def random_schedule(fjsp):
    schedule = []
    for job in fjsp.jobs:
        job_schedule = []
        for operation in job:
            machines, times = operation
            idx = random.randint(0, len(machines) - 1)
            job_schedule.append((machines[idx], times[idx]))
        schedule.append(job_schedule)
    return schedule

# 生成初始化种群
def generate_population(quality, size, fjsp):
    population = []
    for _ in range(size):
        if quality == "heuristic_schedule":
            schedule = heuristic_schedule(fjsp)
        elif quality == "mixed_schedule":
            schedule = mixed_schedule(fjsp)
        else:
            schedule = random_schedule(fjsp)
        population.append(schedule)
    return population

if __name__ == '__main__':
    # 解析dataset数据为parsed_data
    file_path = "../dataset/Dauzere/Text/12a.fjs"
    parsed_data = parse(file_path)
    print(parsed_data)

    # 将parsed_data格式化解析为fjsp自定义对象
    jobs, machines = parse_fjsp_data(parsed_data)
    fjsp = FJSP(jobs, machines)

    # 实验设置
    population_size = 50
    generations = 100
    runs = 20

    # # 验证初始化种群质量
    # quality_metrics = {"high": [], "medium": [], "low": []}
    # for quality in quality_metrics.keys():
    #     for _ in range(runs):
    #         population = generate_population(quality, population_size, fjsp)
    #         makespans = [fjsp.makespan(ind) for ind in population]
    #         quality_metrics[quality].append(np.mean(makespans))
    # 评估初始化种群的质量
    def evaluate_population_quality(population, fjsp):
        makespans = [fjsp.makespan(ind) for ind in population]
        return np.mean(makespans)  # 返回种群的平均 makespan

    # 运行实验，基于各规则生成初始种群
    results = {"heuristic_schedule": [], "mixed_schedule": [], "random_schedule": []}
    for quality in results.keys():
        for _ in range(runs):
            population = generate_population(quality, population_size, fjsp)
            #print(population)
            best_schedule = genetic_algorithm.genetic_algorithm(fjsp, population, generations)
            best_makespan = fjsp.makespan(best_schedule)
            results[quality].append(best_makespan)
        print(f"{quality}方式得到的一组makespan：")
        print(results[quality])
    print("=" * 40)
    print(results)

    # 统计分析
    for quality, metrics in results.items():
        print(f"{quality} population:")
        print(f"  Mean makespan: {np.mean(metrics)}")
        print(f"  Std makespan: {np.std(metrics)}")
        print(f"  Min makespan: {np.min(metrics)}")
        print(f"  Max makespan: {np.max(metrics)}")

    print("=" * 40)

    # 假设检验（t 检验）
    # 分别两两进行 -> 独立性t检验：
    t_stat_1, p_value_1 = ttest_ind(results["heuristic_schedule"], results["mixed_schedule"])
    print(f"  Mean绝对差值: {abs(np.mean(results['heuristic_schedule']) - np.mean(results['mixed_schedule']))}")
    print(f"  T-statistic: {t_stat_1}, P-value: {p_value_1}")
    print("-" * 40)

    t_stat_2, p_value_2 = ttest_ind(results["heuristic_schedule"], results["random_schedule"])
    print(f"  Mean绝对差值: {abs(np.mean(results['heuristic_schedule']) - np.mean(results['random_schedule']))}")
    print(f"  T-statistic: {t_stat_2}, P-value: {p_value_2}")
    print("-" * 40)

    t_stat_3, p_value_3 = ttest_ind(results["mixed_schedule"], results["random_schedule"])
    print(f"  Mean绝对差值: {abs(np.mean(results['mixed_schedule']) - np.mean(results['random_schedule']))}")
    print(f"  T-statistic: {t_stat_3}, P-value: {p_value_3}")

    groups = ['heuristic_schedule', 'mixed_schedule', 'random_schedule']
    means = [np.mean(results['heuristic_schedule']), np.mean(results['mixed_schedule']), np.mean(results['random_schedule'])]
    stds = [np.std(results['heuristic_schedule']), np.std(results['mixed_schedule']), np.std(results['random_schedule'])]
    mins = [np.min(results['heuristic_schedule']), np.min(results['mixed_schedule']), np.min(results['random_schedule'])]
    maxs = [np.max(results['heuristic_schedule']), np.max(results['mixed_schedule']), np.max(results['random_schedule'])]

    # 绘制柱状图
    x = np.arange(len(groups))
    plt.bar(x, means, yerr=stds, capsize=5, color=['blue', 'orange', 'green'])
    plt.xticks(x, groups)
    plt.ylabel('makespan 均值(s)')
    plt.title('初始种群质量指标统计')

    # 标注 Min 和 Max
    for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
        plt.text(i, means[i] + stds[i] + 10, f'Min: {min_val}\nMax: {max_val}', ha='center')

    plt.show()
