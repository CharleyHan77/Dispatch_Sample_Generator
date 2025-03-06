import matplotlib.pyplot as plt
import numpy as np

# 数据
groups = ['heuristic_schedule', 'mixed_schedule', 'random_schedule']
means = [2080.0, 2107.05, 2128.8]
stds = [0.0, 12.29, 19.94]
mins = [2080, 2082, 2095]
maxs = [2080, 2140, 2193]

# 绘制柱状图
x = np.arange(len(groups))
plt.bar(x, means, yerr=stds, capsize=5, color=['blue', 'orange', 'green'])
plt.xticks(x, groups)
plt.ylabel('Mean makespan (s)')
plt.title('Initial Population Statistics')

# 标注 Min 和 Max
for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
    plt.text(i, means[i] + stds[i] + 10, f'Min: {min_val}\nMax: {max_val}', ha='center')

plt.show()
