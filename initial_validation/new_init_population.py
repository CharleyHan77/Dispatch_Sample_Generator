import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser, gantt
from initial_validation.ga_fjsp import ga_new

def find_convergence_point(curve):
    """
    使用Savitzky-Golay滤波和峰值检测找到收敛点
    :param curve: 收敛曲线数据
    :return: 收敛点的索引
    """
    # 使用Savitzky-Golay滤波平滑曲线
    window_length = min(21, len(curve))
    if window_length % 2 == 0:
        window_length += 1
    smoothed_curve = savgol_filter(curve, window_length, 3)
    
    # 计算一阶差分
    diff = np.diff(smoothed_curve)
    
    # 找到差分值开始稳定的点（即收敛点）
    threshold = np.std(diff) * 0.1
    convergence_idx = np.where(np.abs(diff) < threshold)[0][0]
    
    return convergence_idx

def plot_convergence_curves(convergence_data: dict, save_path: str = None):
    """
    绘制收敛曲线
    :param convergence_data: 收敛数据字典
    :param save_path: 保存路径
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6))
    
    for init_method, curves in convergence_data.items():
        # 计算平均收敛曲线
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        
        # 绘制平均曲线
        plt.plot(mean_curve, label=f'{init_method} (mean)')
        
        # 绘制标准差范围
        plt.fill_between(range(len(mean_curve)), 
                        mean_curve - std_curve, 
                        mean_curve + std_curve, 
                        alpha=0.2)
        
        # 找到并标记收敛点
        convergence_idx = find_convergence_point(mean_curve)
        convergence_value = mean_curve[convergence_idx]
        plt.plot(convergence_idx, convergence_value, 'ro')
        # 添加坐标值标注
        plt.annotate(f'({convergence_idx}, {convergence_value:.2f})',
                    xy=(convergence_idx, convergence_value),
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('迭代数')
    plt.ylabel('Makespan')
    plt.title('Brandimarte/Mk01.fjs 三种初始化方式的平均收敛曲线（当前元启发算法最大迭代数100 每种初始化方式运行20次）')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    # 获取数据集文件的绝对路径
    file_path = os.path.join(project_root, "dataset", "Brandimarte", "Mk01.fjs")
    
    # 设置执行次数
    runs = 20
    parameters = parser.parse(file_path)
    print("数据集.fjs文件解析结果：")
    print(parameters)
    print("=" * 40)

    results = {"heuristic": [], "mixed": [], "random": []}
    # 记录每种初始化方法的收敛曲线
    convergence_data = {
        "heuristic": [],
        "mixed": [],
        "random": []
    }
    # 假设执行20次，则每个初始化方法下对应的收敛曲线为20个，每一个curve代表一次执行的收敛曲线列表，比如 curve1 = [83, 78, 78, 73, 68, 63, ... 50, 50, 51, 51, 50, 51, 51, 51, 51, 51, 51]
    # convergence_data = {
    #     "heuristic": [curve1, curve2, ..., curve20],
    #     "mixed": [curve1, curve2, ..., curve20],
    #     "random": [curve1, curve2, ..., curve20]
    # }

    # 记录每种初始化方法的收敛时间
    convergence_times = {
        "heuristic": [],
        "mixed": [],
        "random": []
    }
    
    for init_method in results.keys():
        print("%" * 80)
        print(f"初始化方法：{init_method}")
        for i in range(runs):
            print(f"第{i + 1}次执行")
            # 修改ga_new函数调用，使其返回最终解和收敛曲线
            best_makespan, convergence_curve = ga_new(parameters, init_method, return_convergence=True)
            print("一次执行的收敛曲线:", convergence_curve)
            print(f"GA最终解makespan：{best_makespan}")
            results[init_method].append(best_makespan)
            convergence_data[init_method].append(convergence_curve)
            
            # 计算收敛效率
            convergence_idx = find_convergence_point(convergence_curve)
            convergence_times[init_method].append(convergence_idx)
            print(f"每次迭代的早期收敛代数：{convergence_idx}")
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
        print(f"  平均收敛代数: {np.mean(convergence_times[init_method]):.2f}")
        print(f"  收敛代数标准差: {np.std(convergence_times[init_method]):.2f}")
        print(f"  时间复杂度: O(G * P * N * O * M)")  # G:最大迭代次数, P:种群大小, N:作业数, O:每个作业的最大操作数, M:机器数

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
    
    # 绘制收敛曲线
    plot_convergence_curves(convergence_data, save_path="convergence_curves.png")
