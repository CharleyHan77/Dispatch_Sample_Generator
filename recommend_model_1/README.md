# 初始化策略推荐系统

## 概述

本系统实现了基于多特征融合相似度的初始化策略推荐功能，通过两阶段混合推荐方法为新FJSP数据实例推荐最合适的初始化策略。

## 系统架构

### 两阶段混合推荐

#### 阶段一：候选集生成
- 使用现有的多特征融合相似度检索系统
- 获得Top 5最相似历史样本
- 这些样本对应的初始化方法性能数据构成候选策略池

#### 阶段二：策略推荐
- 基于相似度加权的性能评估
- 综合考虑求解精度和收敛效率
- 输出最优初始化策略推荐

## 特征融合

系统融合了四种特征：
1. **基础特征** (权重: 0.3) - 工件数、机器数、操作数等
2. **加工时间特征** (权重: 0.25) - 加工时间分布特征
3. **KDE特征** (权重: 0.2) - 核密度估计特征
4. **析取图特征** (权重: 0.25) - 基于WL算法的图结构特征

## 性能评估

策略性能评估考虑两个维度：
1. **求解精度** (权重: 0.6) - 基于目标函数值的优化程度
2. **收敛效率** (权重: 0.4) - 基于收敛代数的效率评估

## 使用方法

### 1. 基本使用

```python
from initialization_strategy_recommender import InitializationStrategyRecommender

# 创建推荐器
recommender = InitializationStrategyRecommender()

# 执行推荐
result = recommender.recommend(top_k=5)

# 获取推荐结果
recommended_strategy = result['recommended_strategy']
print(f"推荐策略: {recommended_strategy}")
```

### 2. 自定义配置

```python
# 自定义路径
recommender = InitializationStrategyRecommender(
    dataset_features_path="../output/dataset_features.json",
    init_validity_result_path="../output/init_validity_result",
    new_data_path="../feature_similarity_weighting/new_data_ptr/new_data_features.json"
)

# 自定义权重
recommender.feature_weights = {
    'basic': 0.3,
    'processing_time': 0.25,
    'kde': 0.2,
    'disjunctive_graph': 0.25
}

recommender.performance_weights = {
    'solution_quality': 0.6,
    'convergence_efficiency': 0.4
}
```

### 3. 分阶段执行

```python
# 阶段一：候选集生成
candidates = recommender.stage1_candidate_generation(top_k=5)

# 阶段二：策略推荐
result = recommender.stage2_strategy_recommendation(candidates)
```

## 输出结果

### JSON结果格式

```json
{
  "candidates": [
    {
      "instance": "Brandimarte/Mk01.fjs",
      "similarity": 0.823,
      "strategies": {
        "heuristic": {...},
        "mixed": {...},
        "random": {...}
      }
    }
  ],
  "strategy_recommendations": {
    "heuristic": {
      "final_score": 0.756,
      "weighted_scores": [...],
      "total_similarity_weight": 3.245,
      "candidate_count": 5
    }
  },
  "recommended_strategy": "heuristic",
  "recommendation_summary": {
    "top_strategy": "heuristic",
    "top_score": 0.756,
    "candidate_count": 5,
    "available_strategies": 3
  }
}
```

### 可视化结果

系统会生成包含以下内容的可视化图表：
1. 候选样本相似度分布
2. 策略评分对比
3. 候选样本策略性能热力图
4. 推荐摘要

## 数据要求

### 输入数据格式

1. **历史数据集特征** (`output/dataset_features.json`)
   - 包含基础特征、加工时间特征、KDE特征、析取图特征

2. **初始化策略验证结果** (`output/init_validity_result/`)
   - 每个FJS文件的验证结果JSON
   - 包含不同初始化策略的性能统计

3. **新数据特征** (`feature_similarity_weighting/new_data_*/new_data_features.json`)
   - 新FJS实例的特征数据

### 验证结果JSON格式

```json
{
  "dataset": "Brandimarte",
  "instance": "Mk01.fjs",
  "meta_heuristic": "HA(GA+TS)",
  "execution_times": 20,
  "max_iterations": 100,
  "initialization_methods": {
    "heuristic": {
      "mean": 49.6,
      "std": 2.615,
      "min": 44,
      "max": 55,
      "avg_convergence_generation": 20.55,
      "convergence_generation_std": 4.985
    }
  }
}
```

## 运行示例

```bash
# 直接运行
python initialization_strategy_recommender.py

# 输出结果将保存在 recommendation_results/ 目录下
```

## 扩展功能

### 帕累托前沿选择法

系统预留了帕累托前沿选择法的接口，可以在后续版本中实现：

```python
# 未来版本将支持
recommender.use_pareto_frontier_selection()
```

### 自定义相似度计算

可以扩展或修改相似度计算方法：

```python
# 自定义相似度计算函数
def custom_similarity_function(data1, data2):
    # 实现自定义相似度计算逻辑
    pass

recommender.custom_similarity = custom_similarity_function
```

## 注意事项

1. 确保所有依赖的数据文件路径正确
2. 新数据特征格式需要与历史数据一致
3. 初始化策略验证结果需要包含完整的性能统计信息
4. 权重配置可以根据具体应用场景进行调整

## 依赖项

- numpy
- pandas
- matplotlib
- seaborn
- 现有的特征相似度计算模块 