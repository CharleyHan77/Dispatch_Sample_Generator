# LambdaMART特征描述

## 特征映射 (Feature Mapping)

### 基本特征 (Basic Features) - Feature ID 1-5:
1. num_jobs: 作业数量
2. num_machines: 机器数量  
3. total_operations: 总操作数
4. avg_available_machines: 平均可用机器数
5. std_available_machines: 可用机器数标准差

### 处理时间特征 (Processing Time Features) - Feature ID 6-10:
6. processing_time_mean: 处理时间均值
7. processing_time_std: 处理时间标准差
8. processing_time_min: 最小处理时间
9. processing_time_max: 最大处理时间
10. machine_time_variance: 机器时间方差

### 析取图特征 (Disjunctive Graph Features) - Feature ID 11-12:
11. nodes_count: 节点数量
12. edges_count: 边数量

### KDE特征 (KDE Features) - Feature ID 13-34:
#### KDE统计特征 (13-19):
13. kde_density_mean: KDE密度均值
14. kde_density_std: KDE密度标准差
15. kde_density_min: KDE密度最小值
16. kde_density_max: KDE密度最大值
17. kde_density_median: KDE密度中位数
18. kde_density_q25: KDE密度25%分位数
19. kde_density_q75: KDE密度75%分位数

#### KDE自适应采样特征 (20-29):
20-29: 均匀采样的10个代表性密度值（自适应长度，避免补零）

#### KDE分布形状特征 (30-34):
30. kde_skewness: 分布偏度（衡量分布的对称性）
31. kde_kurtosis: 分布峰度（衡量分布的尖锐程度）
32. kde_peak_position: 峰值位置比例（0-1之间）
33. kde_cv: 变异系数（标准差/均值，衡量离散程度）
34. kde_effective_width: 有效宽度（包含90%概率质量的区间比例）

## 查询映射 (Query Mapping)
- 每个查询(qid) = 一个FJSP实例
- 每个实例对应3个文档(初始化方法): heuristic, mixed, random

## 相关性分数 (Relevance Scores)

### 综合性能评估 (Comprehensive Assessment)
基于多维性能指标的加权评估:

#### 性能维度:
1. **目标值质量** (Objective Quality): 综合最小值、均值和最大值，权重50%最小值+30%均值+20%最大值
2. **目标稳定性** (Objective Stability): 基于标准差，值越小越好
3. **收敛速度** (Convergence Speed): 基于平均收敛代数，越小越好
4. **收敛稳定性** (Convergence Stability): 基于收敛代数标准差，越小越好

#### 权重配置:
- **balanced**: 平衡配置 (40%, 25%, 20%, 15%)
- **quality_focused**: 注重解质量 (60%, 30%, 5%, 5%)
- **speed_focused**: 注重收敛速度 (30%, 20%, 35%, 15%)
- **objective_stability_focused**: 注重目标稳定性 (30%, 50%, 15%, 5%)
- **convergence_stability_focused**: 注重收敛稳定性 (30%, 5%, 15%, 50%)

### 分数模式:
- **排名模式**: 0(最差), 1(中等), 2(最好)
- **连续模式**: 基于加权综合分数的连续值

### 单一指标模式 (对比基准):
- mean: 基于平均makespan
- min: 基于最小makespan

## 元数据 (Metadata in Comments)
- instance: FJS文件路径
- method: 初始化方法
- meta_heuristic: 元启发式算法
- execution_times: 执行次数
- max_iterations: 最大迭代次数
