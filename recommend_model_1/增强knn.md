# FJSP初始化策略推荐系统 - 增强型KNN算法详解

## 📋 目录
- [系统概述](#系统概述)
- [核心算法架构](#核心算法架构)
- [KNN方法详解](#knn方法详解)
- [权重配置体系](#权重配置体系)
- [两阶段推荐流程](#两阶段推荐流程)
- [技术实现细节](#技术实现细节)
- [性能评估体系](#性能评估体系)
- [使用指南](#使用指南)

---

## 🎯 系统概述

### 核心思想
基于**增强型KNN算法**的两阶段推荐系统，专门为柔性作业车间调度问题（FJSP）提供初始化策略智能推荐。该系统将传统KNN方法扩展为多特征融合、加权相似度计算的高级推荐算法。

### 主要特点
- ✅ **多维特征融合**：4种特征类型，8个权重维度
- ✅ **两阶段推荐**：相似度检索 + 策略推荐
- ✅ **加权KNN**：基于相似度的加权聚合
- ✅ **多维性能评估**：4个性能指标综合评分
- ✅ **自适应权重**：细化权重配置系统

---

## 🧠 核心算法架构

### 算法分类
本系统本质上是一个**增强型KNN算法**，具体表现为：

```
传统KNN → 增强型KNN
    ↓         ↓
单特征     多特征融合
等权重     加权相似度  
简单投票   两阶段推荐
固定K值    性能驱动选择
```

### 与传统KNN的对比

| 特征 | 传统KNN | 本系统（增强型KNN） |
|------|---------|---------------------|
| **距离度量** | 单一欧氏距离 | 4种特征融合相似度 |
| **权重机制** | 等权重 | 细化权重配置 |
| **邻居选择** | 固定K值 | 两阶段动态选择 |
| **预测方式** | 简单投票 | 相似度加权性能评分 |
| **特征处理** | 原始特征 | 标准化+多维度特征 |

---

## 🎯 KNN方法详解

### 1. 相似度计算（距离度量）

#### 1.1 基础特征相似度
```python
# 欧氏距离计算
def calculate_euclidean_distance(features1, features2):
    vec1 = np.array(list(features1.values()))
    vec2 = np.array(list(features2.values()))
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# 距离转相似度
def normalize_distance(distance, max_distance):
    return 1 - (distance / max_distance)
```

#### 1.2 KDE特征相似度（JS散度）
```python
def calculate_js_divergence(p, q):
    """Jensen-Shannon散度计算概率分布相似度"""
    p, q = p / np.sum(p), q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m + 1e-10))
    kl_qm = np.sum(q * np.log2(q / m + 1e-10))
    return 0.5 * (kl_pm + kl_qm)
```

#### 1.3 析取图相似度（结构相似度）
```python
def calculate_disjunctive_graph_similarity(graph1, graph2):
    """基于WL算法的图结构相似度"""
    # 结构相似度：节点数和边数的余弦相似度
    structure_similarity = cosine_similarity(graph1_struct, graph2_struct)
    
    # 标签相似度：WL算法生成的标签分布
    label_similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
    
    # 综合相似度
    return 0.5 * structure_similarity + 0.5 * label_similarity
```

### 2. KNN邻居查找

#### 2.1 阶段一：Top-K相似样本检索
```python
def stage_one_similarity_search(new_data_features, top_k=5):
    """KNN第一阶段：查找K个最相似邻居"""
    similarity_results = {}
    
    # 计算与所有历史样本的相似度
    for fjs_path in self.normalized_features.keys():
        similarity = self.calculate_weighted_similarity(new_data, fjs_path)
        similarity_results[fjs_path] = similarity
    
    # 按相似度排序，返回Top-K
    sorted_results = sorted(similarity_results.items(), 
                          key=lambda x: x[1]["weighted_similarity"], 
                          reverse=True)
    
    return sorted_results[:top_k]
```

### 3. 加权预测

#### 3.1 相似度加权聚合
```python
def stage_two_strategy_recommendation(candidate_samples, top_k=3):
    """KNN第二阶段：基于邻居进行加权预测"""
    strategy_scores = {}
    
    for strategy_name, performances in strategy_performance.items():
        total_weighted_score = 0
        total_weight = 0
        
        # 使用相似度作为权重进行加权平均
        for perf in performances:
            weight = perf['similarity_score']  # KNN权重
            total_weighted_score += weight * perf['performance_score']
            total_weight += weight
        
        weighted_avg_score = total_weighted_score / total_weight
        strategy_scores[strategy_name] = weighted_avg_score
    
    return sorted(strategy_scores.items(), reverse=True)[:top_k]
```

---

## ⚖️ 权重配置体系

### 1. 特征层级权重

#### 1.1 基础特征权重（总权重：30%）
```python
'basic_features': {
    'num_jobs': 0.08,                    # 工件数量 (8%)
    'num_machines': 0.08,                # 机器数量 (8%)
    'total_operations': 0.06,            # 总操作数 (6%)
    'avg_available_machines': 0.05,      # 平均可用机器数 (5%)
    'std_available_machines': 0.03       # 可用机器数标准差 (3%)
}
# 小计：30%
```

#### 1.2 加工时间特征权重（总权重：25%）
```python
'processing_time_features': {
    'processing_time_mean': 0.08,        # 平均加工时间 (8%)
    'processing_time_std': 0.06,         # 加工时间标准差 (6%)
    'processing_time_min': 0.04,         # 最小加工时间 (4%)
    'processing_time_max': 0.04,         # 最大加工时间 (4%)
    'machine_time_variance': 0.03        # 机器时间方差 (3%)
}
# 小计：25%
```

#### 1.3 高级特征权重
```python
'kde_similarity_weight': 0.2,           # KDE特征权重 (20%)
'disjunctive_similarity_weight': 0.25   # 析取图特征权重 (25%)
```

### 2. 综合相似度计算公式

```python
weighted_similarity = (
    basic_detailed_similarity +          # 30% (基础特征细化权重和)
    processing_detailed_similarity +     # 25% (加工时间特征细化权重和)
    0.2 * kde_similarity +              # 20% (KDE相似度)
    0.25 * disjunctive_similarity       # 25% (析取图相似度)
)
```

### 3. 细化权重计算示例

#### 3.1 基础特征细化相似度
```python
basic_detailed_similarity = 0
for feature_name, weight in basic_features_weights.items():
    distance = abs(new_feature[feature_name] - hist_feature[feature_name])
    feature_similarity = np.exp(-distance**2 / 2)  # 高斯相似度函数
    basic_detailed_similarity += weight * feature_similarity

# 结果范围：[0, 0.30]
```

---

## 🔄 两阶段推荐流程

### 阶段一：多特征相似度检索（KNN邻居查找）

```
新FJSP实例
    ↓
特征提取
    ↓
┌─────────────────────────────────────────────────────────┐
│ 基础特征        │ 加工时间特征  │ KDE特征    │ 析取图特征 │
│ num_jobs等      │ mean, std等   │ 概率密度   │ 图结构    │
└─────────────────────────────────────────────────────────┘
    ↓           ↓           ↓           ↓
欧氏距离      欧氏距离     JS散度      WL算法
    ↓           ↓           ↓           ↓
权重0.30      权重0.25    权重0.20    权重0.25
    ↓
加权融合相似度
    ↓
Top-K最相似样本 (KNN邻居)
```

### 阶段二：策略推荐（加权预测）

```
Top-K相似样本
    ↓
提取策略性能数据
    ↓
┌─────────────────────────────────────────────────────────┐
│ Random策略   │ Heuristic策略  │ Mixed策略              │
│ 性能数据     │ 性能数据       │ 性能数据               │
└─────────────────────────────────────────────────────────┘
    ↓
计算多维度性能评分
    ↓
┌─────────────────────────────────────────────────────────┐
│ Makespan     │ 收敛速度      │ 稳定性      │ 收敛稳定性│
│ 权重40%      │ 权重25%       │ 权重20%     │ 权重15%   │
└─────────────────────────────────────────────────────────┘
    ↓
相似度加权聚合
    ↓
Top-K推荐策略
```

---

## 🛠️ 技术实现细节

### 1. 数据预处理

#### 1.1 特征标准化
```python
def normalize_features(features_dict):
    """Z-score标准化，确保不同特征在同一量纲"""
    normalized_features = {}
    for feature_type in ['basic_features', 'processing_time_features']:
        # 提取所有样本的特征值
        all_values = {}
        for fjs_path, features in features_dict.items():
            for key, value in features[feature_type].items():
                if key not in all_values:
                    all_values[key] = []
                all_values[key].append(value)
        
        # 计算均值和标准差进行标准化
        for key, values in all_values.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            # Z-score标准化：(x - μ) / σ
```

#### 1.2 距离归一化
```python
def normalize_distance(distance, max_distance):
    """将距离归一化到[0,1]，转换为相似度"""
    if max_distance <= 0:
        return 1.0
    return 1 - (distance / max_distance)
```

### 2. 相似度计算优化

#### 2.1 高斯相似度函数
```python
# 用于细化特征相似度计算
def gaussian_similarity(distance):
    return np.exp(-distance**2 / 2)
```

#### 2.2 余弦相似度
```python
# 用于析取图结构相似度
def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)
```

---

## 📊 性能评估体系

### 1. 多维度性能指标

#### 1.1 Makespan评分（权重：40%）
```python
makespan_score = 1.0 / (1.0 + mean_makespan / 1000.0)
```
- **目标**：最小化完工时间
- **评分逻辑**：makespan越小，评分越高
- **权重理由**：完工时间是调度问题的主要优化目标

#### 1.2 收敛速度评分（权重：25%）
```python
max_iterations = 100
convergence_speed_score = 1.0 - (avg_convergence_gen / max_iterations)
convergence_speed_score = max(0.0, min(1.0, convergence_speed_score))
```
- **目标**：快速收敛到最优解
- **评分逻辑**：收敛代数越少，评分越高
- **权重理由**：收敛速度影响算法实用性

#### 1.3 稳定性评分（权重：20%）
```python
stability_score = 1.0 / (1.0 + std_makespan / 10.0)
```
- **目标**：结果稳定可靠
- **评分逻辑**：makespan标准差越小，评分越高
- **权重理由**：稳定性确保算法可靠性

#### 1.4 收敛稳定性评分（权重：15%）
```python
convergence_stability_score = 1.0 / (1.0 + convergence_std / 10.0)
```
- **目标**：收敛过程稳定
- **评分逻辑**：收敛代数标准差越小，评分越高
- **权重理由**：收敛稳定性影响算法鲁棒性

### 2. 综合性能评分
```python
performance_score = (
    0.4 * makespan_score +                    # 40%
    0.25 * convergence_speed_score +          # 25%
    0.2 * stability_score +                   # 20%
    0.15 * convergence_stability_score        # 15%
)
```

---

## 📚 使用指南

### 1. 基本使用

#### 1.1 命令行接口
```bash
# 基本推荐
python initialization_strategy_recommender.py new_data.fjs

# 自定义参数
python initialization_strategy_recommender.py new_data.fjs \
  --top-k-similar 3 \
  --top-k-strategies 2 \
  --output-dir my_results

# 使用自定义权重配置
python initialization_strategy_recommender.py new_data.fjs \
  --weights-config custom_weights.json
```

#### 1.2 参数说明
- `fjs_file`：输入的FJS文件路径（必需）
- `--top-k-similar`：阶段一返回的最相似样本数量（默认：5）
- `--top-k-strategies`：阶段二推荐的策略数量（默认：3）
- `--output-dir`：输出目录（默认：result/recommender_output）
- `--weights-config`：自定义权重配置文件路径

### 2. 输出结果解读

#### 2.1 阶段一结果（KNN邻居）
```
Top 5 最相似的历史样本:
1. dataset/BehnkeGeiger/Behnke29.fjs
   综合加权相似度: 0.8542
   基础特征相似度: 0.8234
   加工时间特征相似度: 0.8765
   KDE相似度: 0.8123
   析取图相似度: 0.8901
```

#### 2.2 阶段二结果（策略推荐）
```
Top 3 推荐策略:
1. heuristic
   加权性能评分: 0.7234
   详细评分:
     Makespan评分: 0.7456
     收敛速度评分: 0.7123
     稳定性评分: 0.7234
     收敛稳定性评分: 0.6987
```

### 3. 自定义权重配置

#### 3.1 权重配置文件格式（JSON）
```json
{
  "weights": {
    "basic_features": {
      "num_jobs": 0.1,
      "num_machines": 0.1,
      "total_operations": 0.05,
      "avg_available_machines": 0.03,
      "std_available_machines": 0.02
    },
    "processing_time_features": {
      "processing_time_mean": 0.1,
      "processing_time_std": 0.08,
      "processing_time_min": 0.04,
      "processing_time_max": 0.04,
      "machine_time_variance": 0.04
    },
    "kde_similarity_weight": 0.25,
    "disjunctive_similarity_weight": 0.25
  }
}
```

---

## 🎯 算法优势与特点

### 1. 相比传统KNN的优势

| 方面 | 传统KNN | 本系统 |
|------|---------|--------|
| **特征处理** | 单维度原始特征 | 多维度融合特征 |
| **距离度量** | 欧氏距离 | 四种相似度融合 |
| **权重机制** | 无权重或等权重 | 细化权重配置 |
| **预测方式** | 简单投票/平均 | 相似度加权性能评分 |
| **适应性** | 固定算法 | 领域特定优化 |

### 2. 系统核心特点

#### 2.1 多特征融合
- ✅ **基础特征**：工件数、机器数等结构特征
- ✅ **时间特征**：加工时间分布特征  
- ✅ **概率特征**：KDE概率密度特征
- ✅ **图特征**：析取图结构特征

#### 2.2 智能权重分配
- ✅ **层级权重**：特征类别级权重 + 细粒度权重
- ✅ **领域知识**：基于FJSP特点的权重设计
- ✅ **可配置性**：支持自定义权重配置

#### 2.3 两阶段优化
- ✅ **阶段一**：高效的相似度检索（KNN邻居查找）
- ✅ **阶段二**：基于性能的智能推荐（加权预测）
- ✅ **性能驱动**：不仅考虑相似度，还考虑实际性能

---

## 📈 权重配置详细分析

### 1. 特征权重设计原理

#### 1.1 基础特征权重分布（总计30%）
```python
基础特征权重分析：
├── num_jobs (8%)           # 工件数量 - 影响问题复杂度
├── num_machines (8%)       # 机器数量 - 影响调度灵活性  
├── total_operations (6%)   # 总操作数 - 反映工作负载
├── avg_available_machines (5%)  # 平均可用机器 - 柔性程度
└── std_available_machines (3%)  # 可用机器标准差 - 柔性变异
```

**设计理念**：
- 工件数和机器数权重最高（各8%），因为它们是决定问题规模的核心因素
- 总操作数次之（6%），反映整体工作负载
- 柔性相关指标权重较低，作为补充信息

#### 1.2 加工时间特征权重分布（总计25%）
```python
加工时间特征权重分析：
├── processing_time_mean (8%)     # 平均加工时间 - 总体时间水平
├── processing_time_std (6%)      # 时间标准差 - 时间分布离散性
├── processing_time_min (4%)      # 最小时间 - 时间下界
├── processing_time_max (4%)      # 最大时间 - 时间上界  
└── machine_time_variance (3%)    # 机器时间方差 - 机器间差异
```

**设计理念**：
- 平均时间权重最高（8%），反映整体时间水平
- 标准差次之（6%），体现时间分布特征
- 边界值权重中等（各4%），提供时间范围信息
- 机器方差权重最低（3%），作为辅助指标

#### 1.3 高级特征权重分布（共45%）
```python
高级特征权重分析：
├── kde_similarity_weight (20%)        # KDE相似度 - 概率分布特征
└── disjunctive_similarity_weight (25%) # 析取图相似度 - 结构特征
```

**设计理念**：
- 析取图权重最高（25%），因为图结构最能反映调度问题的内在特征
- KDE权重次之（20%），概率分布特征提供统计信息
- 两者合计45%，体现了高级特征的重要性

### 2. 性能评估权重设计（策略推荐阶段）

#### 2.1 性能指标权重分布
```python
性能评估权重分析：
├── makespan (40%)                    # 完工时间 - 主要优化目标
├── convergence_speed (25%)           # 收敛速度 - 算法效率
├── stability (20%)                   # 结果稳定性 - 算法可靠性
└── convergence_stability (15%)       # 收敛稳定性 - 过程一致性
```

**设计理念**：
- Makespan权重最高（40%），是调度问题的核心优化目标
- 收敛速度次之（25%），关乎算法的实用性
- 稳定性指标合计35%，确保算法的可靠性和一致性

### 3. 权重设计的FJSP领域特色

#### 3.1 领域知识融入
```python
FJSP特征权重设计考虑：
1. 问题规模特征（工件数、机器数）→ 高权重
2. 柔性程度特征（可用机器分布）→ 中等权重  
3. 时间特征（加工时间分布）→ 中等权重
4. 结构特征（析取图）→ 高权重
5. 统计特征（KDE）→ 中等权重
```

#### 3.2 权重平衡原则
- **结构优先**：析取图特征权重最高（25%），体现问题结构重要性
- **规模次之**：基础特征总权重30%，反映问题规模影响
- **时间补充**：时间特征总权重25%，提供执行层面信息
- **统计辅助**：KDE特征权重20%，作为统计补充

---

## 📊 KNN算法实现的技术创新

### 1. 距离度量创新

#### 1.1 多种距离度量融合
```python
距离度量体系：
├── 欧氏距离（基础特征、时间特征）
│   └── 适用于数值型特征的相似度计算
├── Jensen-Shannon散度（KDE特征）  
│   └── 适用于概率分布的相似度计算
└── 余弦相似度 + Jaccard系数（析取图特征）
    └── 适用于图结构的相似度计算
```

#### 1.2 相似度标准化策略
```python
def normalize_distance(distance, max_distance):
    """统一的相似度标准化方法"""
    # 将所有距离都转换为[0,1]范围内的相似度
    # 0表示完全不相似，1表示完全相似
    if max_distance <= 0:
        return 1.0
    return 1 - (distance / max_distance)
```

### 2. 权重学习机制

#### 2.1 高斯相似度函数
```python
def gaussian_similarity(distance):
    """高斯核函数计算特征相似度"""
    # 使用高斯函数将距离映射为相似度
    # 距离越小，相似度越接近1
    # 距离越大，相似度指数衰减到0
    return np.exp(-distance**2 / 2)
```

#### 2.2 细化权重聚合
```python
def calculate_detailed_similarity():
    """细化权重的特征聚合"""
    detailed_similarity = 0
    for feature_name, weight in feature_weights.items():
        # 单特征相似度计算
        feature_distance = abs(new_feature - hist_feature)
        feature_similarity = gaussian_similarity(feature_distance)
        
        # 权重聚合
        detailed_similarity += weight * feature_similarity
    
    return detailed_similarity
```

### 3. 两阶段KNN实现

#### 3.1 阶段一：增强的K邻居搜索
```python
class EnhancedKNNSearch:
    """增强型KNN搜索"""
    
    def find_k_neighbors(self, query, k=5):
        """多特征融合的K邻居搜索"""
        similarities = []
        
        for candidate in self.dataset:
            # 多特征相似度计算
            basic_sim = self.basic_similarity(query, candidate)
            time_sim = self.time_similarity(query, candidate)  
            kde_sim = self.kde_similarity(query, candidate)
            graph_sim = self.graph_similarity(query, candidate)
            
            # 加权融合
            weighted_sim = (
                0.30 * basic_sim +
                0.25 * time_sim + 
                0.20 * kde_sim +
                0.25 * graph_sim
            )
            
            similarities.append((candidate, weighted_sim))
        
        # 返回Top-K最相似的邻居
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
```

#### 3.2 阶段二：性能驱动的加权预测
```python
class PerformanceDrivenPrediction:
    """性能驱动的KNN预测"""
    
    def weighted_prediction(self, neighbors, k=3):
        """基于邻居性能的加权预测"""
        strategy_scores = {}
        
        for strategy in ['random', 'heuristic', 'mixed']:
            weighted_score = 0
            total_weight = 0
            
            for neighbor, similarity in neighbors:
                # 获取邻居的策略性能
                performance = neighbor.get_strategy_performance(strategy)
                
                # 计算多维度性能评分
                perf_score = self.calculate_performance_score(performance)
                
                # 相似度加权
                weighted_score += similarity * perf_score
                total_weight += similarity
            
            # 加权平均
            if total_weight > 0:
                strategy_scores[strategy] = weighted_score / total_weight
        
        # 返回Top-K策略
        return sorted(strategy_scores.items(), 
                     key=lambda x: x[1], reverse=True)[:k]
```

---

## 🔧 系统优化与扩展

### 1. 性能优化策略

#### 1.1 计算优化
- **特征预计算**：标准化特征预先计算和缓存
- **相似度缓存**：相同特征组合的相似度结果缓存
- **并行计算**：多线程/多进程计算相似度
- **近似算法**：大数据集使用LSH等近似KNN算法

#### 1.2 内存优化
- **增量加载**：按需加载历史数据
- **特征压缩**：使用PCA等降维技术
- **稀疏存储**：稀疏矩阵存储相似度数据

### 2. 算法扩展方向

#### 2.1 自适应权重学习
```python
class AdaptiveWeightLearning:
    """自适应权重学习"""
    
    def learn_weights(self, validation_data):
        """基于验证数据学习最优权重"""
        from scipy.optimize import minimize
        
        def objective(weights):
            # 计算验证集上的预测准确率
            accuracy = self.evaluate_with_weights(weights, validation_data)
            return -accuracy  # 最小化负准确率
        
        # 权重约束：和为1，非负
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # 优化权重
        result = minimize(objective, initial_weights, constraints=constraints)
        return result.x
```

#### 2.2 在线学习机制
```python
class OnlineLearningKNN:
    """在线学习KNN"""
    
    def update_with_feedback(self, query, predicted_strategy, actual_performance):
        """根据反馈更新模型"""
        # 1. 更新数据集
        self.add_sample(query, predicted_strategy, actual_performance)
        
        # 2. 增量更新权重
        self.update_weights_incrementally()
        
        # 3. 更新相似度缓存
        self.invalidate_similarity_cache(query)
```

---

## 📈 总结

这个FJSP初始化策略推荐系统是一个**高度优化和领域特定的KNN算法变种**，主要创新点包括：

### 🎯 核心贡献

1. **多特征融合的KNN**：
   - 将传统单一距离度量扩展为4种特征类型的融合
   - 每种特征使用最适合的相似度计算方法
   - 通过细化权重实现精确的特征重要性控制

2. **两阶段推荐机制**：
   - 阶段一：基于多特征相似度的K邻居搜索
   - 阶段二：基于邻居性能的加权策略推荐
   - 结合相似度和性能双重考量

3. **领域知识集成**：
   - 权重设计融入FJSP领域专业知识
   - 多维度性能评估体系反映调度问题特点
   - 析取图结构特征捕获问题本质

4. **高度可配置性**：
   - 支持自定义权重配置
   - 可调节K值参数
   - 模块化设计便于扩展

### 🚀 技术价值

- **理论价值**：将KNN算法成功扩展到复杂的多特征融合场景
- **实用价值**：为FJSP问题提供了实用的初始化策略推荐工具
- **扩展价值**：框架具有良好的可扩展性，可适用于其他调度问题

这个系统证明了传统机器学习算法（KNN）通过适当的改进和领域知识融入，可以在特定领域中发挥强大的实用价值。

---

*文档版本：v1.0*  
*创建时间：2024年*  
*系统类型：增强型KNN推荐算法*  
*应用领域：柔性作业车间调度问题（FJSP）*

