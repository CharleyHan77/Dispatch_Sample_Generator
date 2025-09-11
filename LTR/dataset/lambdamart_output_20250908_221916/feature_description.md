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
#### KDE有效密度统计特征 (13-19):
- **过滤策略**: 只使用≥最大值1%的有效密度值，避免噪声影响
13. effective_density_mean: 有效密度均值
14. effective_density_std: 有效密度标准差
15. effective_density_min: 有效密度最小值
16. effective_density_max: 有效密度最大值
17. effective_density_median: 有效密度中位数
18. effective_density_q25: 有效密度25%分位数
19. effective_density_q75: 有效密度75%分位数

#### KDE智能采样特征 (20-29):
- **采样策略**: 重点采样高密度区域，而非均匀采样
20-24: 前5个最高峰值（按密度值排序）
25-29: 中等密度区域的5个均匀采样值
- **补充策略**: 当样本不足时，使用统计衍生值而非零值填充

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
