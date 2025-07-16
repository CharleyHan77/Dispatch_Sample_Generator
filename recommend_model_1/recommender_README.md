# 初始化策略推荐系统详细说明

## 系统概述

初始化策略推荐系统是一个基于多特征融合相似度计算的两阶段推荐系统，专门为柔性作业车间调度问题（FJSP）的初始化策略选择提供智能推荐。

## 核心架构

### 两阶段推荐流程

#### 阶段一：多特征相似度检索
基于四种特征融合的加权相似度计算，从历史数据集中检索与新数据最相似的候选样本。

#### 阶段二：策略推荐
基于候选样本的策略性能数据和相似度权重，计算每种初始化策略的综合评分并排序推荐。

## 详细技术规格

### 1. 特征提取与标准化

#### 1.1 基础特征 (权重: 30%)
- **num_jobs**: 作业数量
- **num_machines**: 机器数量  
- **total_operations**: 总操作数
- **avg_available_machines**: 平均可用机器数
- **std_available_machines**: 可用机器数标准差

#### 1.2 加工时间特征 (权重: 25%)
- **processing_time_mean**: 加工时间均值
- **processing_time_std**: 加工时间标准差
- **processing_time_min**: 加工时间最小值
- **processing_time_max**: 加工时间最大值
- **machine_time_variance**: 机器时间方差

#### 1.3 KDE特征 (权重: 20%)
- **density**: 核密度估计的概率密度分布
- 使用Jensen-Shannon散度计算相似度

#### 1.4 析取图特征 (权重: 25%)
- **node_features**: 节点特征统计
- **edge_features**: 边特征统计
- **graph_metrics**: 图结构指标
- 基于Weisfeiler-Lehman图核算法计算相似度

### 2. 相似度计算算法

#### 2.1 欧几里得距离计算
```python
def calculate_euclidean_distance(self, features1, features2):
    """计算两个特征向量的欧几里得距离"""
    return np.sqrt(np.sum((np.array(list(features1.values())) - 
                          np.array(list(features2.values()))) ** 2))
```

#### 2.2 Jensen-Shannon散度
```python
def calculate_js_divergence(self, p, q):
    """计算两个概率分布的Jensen-Shannon散度"""
    # 确保概率分布有效
    p = np.array(p)
    q = np.array(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 计算中间分布
    m = 0.5 * (p + q)
    
    # 计算KL散度
    kl_pm = np.sum(p * np.log2(p / m + 1e-10))
    kl_qm = np.sum(q * np.log2(q / m + 1e-10))
    
    return 0.5 * (kl_pm + kl_qm)
```

#### 2.3 距离归一化
```python
def normalize_distance(self, distance, max_distance):
    """将距离归一化到[0,1]范围，转换为相似度"""
    if max_distance == 0:
        return 1.0
    return 1.0 - (distance / max_distance)
```

### 3. 综合加权相似度计算

```python
weighted_similarity = (
    0.3 * basic_similarity +           # 基础特征权重
    0.25 * processing_similarity +     # 加工时间特征权重
    0.2 * kde_similarity +            # KDE特征权重
    0.25 * disjunctive_similarity     # 析取图特征权重
)
```

### 4. 策略性能评分体系

#### 4.1 多维度性能指标

##### Makespan评分 (权重: 40%)
```python
makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
```
- 目标：最小化完工时间
- 评分越高表示makespan越小，性能越好

##### 收敛速度评分 (权重: 25%)
```python
max_iterations = 100
convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
```
- 目标：快速收敛
- 评分越高表示收敛代数越少，收敛越快

##### 稳定性评分 (权重: 20%)
```python
stability_score = 1.0 / (1.0 + std_makespan / 10.0)
```
- 目标：结果稳定
- 评分越高表示makespan标准差越小，结果越稳定

##### 收敛稳定性评分 (权重: 15%)
```python
convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
```
- 目标：收敛过程稳定
- 评分越高表示收敛代数标准差越小，收敛过程越稳定

#### 4.2 综合性能评分计算

```python
performance_score = (
    weights['makespan'] * makespan_score +
    weights['convergence_speed'] * convergence_speed_score +
    weights['stability'] * stability_score +
    weights['convergence_stability'] * convergence_stability_score
)
```

### 5. 相似度加权策略推荐

#### 5.1 加权平均计算
```python
total_weighted_score = 0
total_weight = 0

for perf in performances:
    weight = perf['similarity_score']  # 使用相似度作为权重
    total_weighted_score += weight * perf['performance_score']
    total_weight += weight

weighted_avg_score = total_weighted_score / total_weight
```

#### 5.2 策略排序与推荐
- 按加权平均评分降序排列
- 返回Top-K推荐策略
- 支持三种初始化策略：heuristic、mixed、random

## 系统使用流程

### 1. 命令行接口

```bash
# 基本使用
python initialization_strategy_recommender.py new_data.fjs

# 自定义参数
python initialization_strategy_recommender.py new_data.fjs --top-k-similar 3 --top-k-strategies 2

# 自定义输出目录
python initialization_strategy_recommender.py new_data.fjs --output-dir my_results
```

### 2. 参数说明

- **fjs_file**: 输入的FJS文件路径（必需）
- **--top-k-similar**: 阶段一返回的最相似样本数量（默认: 5）
- **--top-k-strategies**: 阶段二推荐的策略数量（默认: 3）
- **--output-dir**: 输出目录（默认: result/recommender_output）

### 3. 输出结构

```
result/recommender_output/
└── {fjs_basename}_{timestamp}/
    ├── recommendation_results.json    # 推荐结果
    ├── recommendation_log.log         # 详细日志
    └── visualization/
        ├── similarity_comparison.png   # 相似度对比图
        └── strategy_recommendation.png # 策略推荐图
```

## 输出结果示例

### 1. 推荐结果JSON结构

```json
{
    "timestamp": "2024-01-01T12:00:00",
    "execution_time": 2.5,
    "stage_one_results": {
        "candidate_samples": [
            {
                "fjs_path": "dataset/BehnkeGeiger/Behnke29.fjs",
                "similarity_score": 0.8542,
                "similarity_details": {
                    "basic_similarity": 0.8234,
                    "processing_similarity": 0.8765,
                    "kde_similarity": 0.8123,
                    "disjunctive_similarity": 0.8901,
                    "weighted_similarity": 0.8542
                }
            }
        ]
    },
    "stage_two_results": {
        "recommended_strategies": [
            {
                "strategy_name": "heuristic",
                "weighted_score": 0.7234
            },
            {
                "strategy_name": "mixed", 
                "weighted_score": 0.6891
            },
            {
                "strategy_name": "random",
                "weighted_score": 0.6543
            }
        ]
    }
}
```

### 2. 控制台输出示例

```
================================================================================
推荐结果摘要
================================================================================

阶段一：最相似的历史数据
1. dataset/BehnkeGeiger/Behnke29.fjs
   综合相似度: 0.8542
   基础特征相似度: 0.8234
   加工时间相似度: 0.8765
   KDE相似度: 0.8123
   析取图相似度: 0.8901

阶段二：推荐策略
1. heuristic
   加权性能评分: 0.7234
   详细评分:
     Makespan评分: 0.7456
     收敛速度评分: 0.7123
     稳定性评分: 0.7234
     收敛稳定性评分: 0.6987
```

## 技术特点

### 1. 多特征融合
- 结合基础特征、加工时间特征、KDE特征和析取图特征
- 使用加权融合提高相似度计算的准确性

### 2. 两阶段推荐
- 阶段一：基于特征相似度的候选样本检索
- 阶段二：基于性能数据的策略推荐

### 3. 多维度性能评估
- 考虑makespan、收敛速度、稳定性和收敛稳定性四个维度
- 使用加权评分体系平衡不同性能指标

### 4. 相似度加权
- 使用相似度作为权重，相似度越高的历史样本对推荐结果影响越大
- 确保推荐结果与历史数据的相关性

### 5. 可视化支持
- 自动生成相似度对比图表
- 生成策略推荐评分图表
- 便于结果分析和展示

## 注意事项

1. **数据依赖**: 系统依赖标记数据集 `labeled_dataset/labeled_fjs_dataset.json`
2. **特征提取**: 需要确保新数据能够成功提取所有四种特征
3. **权重调整**: 可根据实际需求调整特征权重和性能指标权重
4. **候选样本数量**: Top-K参数影响推荐精度和计算效率
5. **输出目录**: 系统会自动创建时间戳子目录避免结果覆盖

## 扩展性

系统设计具有良好的扩展性：
- 可添加新的特征类型
- 可调整相似度计算算法
- 可增加新的性能指标
- 可支持更多初始化策略类型 