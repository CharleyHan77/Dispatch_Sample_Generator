import numpy as np
from scipy.stats import ttest_ind
from initial_validation.utils import parser, gantt

from initial_validation.ga_fjsp import ga_new

if __name__ == '__main__':
    file_path = "../dataset/Brandimarte/Text/Mk01.fjs"
    # 设置执行次数
    runs = 20
    parameters = parser.parse(file_path)
    print("数据集.fjs文件解析结果：")
    print(parameters)
    print("=" * 40)

    results = {"heuristic": [], "mixed": [], "random": []}
    for init_method in results.keys():
        print("%" * 80)
        print(f"初始化方法：{init_method}")
        for i in range(runs):
            #print(f"第{i + 1}次执行的初始种群")
            best_makespan = ga_new(parameters, init_method)
            print(f"GA最终解makespan：{best_makespan}")
            results[init_method].append(best_makespan)
            print("="*30)
        print(f"{init_method}方式得到的一组makespan：")
        print(results[init_method])
    print("%" * 80)

    # 输出统计结果及收敛效率相关结果
    for init_method, metrics in results.items():
        print(f"{init_method} 种群:")
        print(f"  Mean makespan: {np.mean(metrics)}")
        print(f"  Std makespan: {np.std(metrics)}")
        print(f"  Min makespan: {np.min(metrics)}")
        print(f"  Max makespan: {np.max(metrics)}")

    print("=" * 40)

    # 假设检验（t 检验）
    # 分别两两进行 -> 独立性t检验：
    t_stat_1, p_value_1 = ttest_ind(results["heuristic"], results["mixed"])
    print("t检验 heuristic 与 mixed")
    print(f"  Mean绝对差值: {abs(np.mean(results['heuristic']) - np.mean(results['mixed']))}")
    print(f"  T-statistic: {t_stat_1}, P-value: {p_value_1}")
    print("-" * 40)

    t_stat_2, p_value_2 = ttest_ind(results["heuristic"], results["random"])
    print("t检验 heuristic 与 random")
    print(f"  Mean绝对差值: {abs(np.mean(results['heuristic']) - np.mean(results['random']))}")
    print(f"  T-statistic: {t_stat_2}, P-value: {p_value_2}")
    print("-" * 40)

    t_stat_3, p_value_3 = ttest_ind(results["mixed"], results["random"])
    print("t检验 mixed 与 random")
    print(f"  Mean绝对差值: {abs(np.mean(results['mixed']) - np.mean(results['random']))}")
    print(f"  T-statistic: {t_stat_3}, P-value: {p_value_3}")