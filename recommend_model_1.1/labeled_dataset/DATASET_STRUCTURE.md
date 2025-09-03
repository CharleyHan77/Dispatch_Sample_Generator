# 调度样本数据结构说明文档

## 概述

本文档详细说明了 `analyze_and_convert.py` 脚本输出的调度样本数据集 `converted_fjs_dataset_new.json` 的数据结构。该数据集是基于原始的 `labeled_fjs_dataset.json` 转换而来，主要变化是将每个原始样本按照不同的初始化方法（heuristic、mixed、random）拆分为三个独立的样本。

## 数据集基本信息

- **文件名**: `converted_fjs_dataset_new.json`
- **总样本数**: 1,188 个样本
- **初始化方法分布**:
  - `heuristic`: 396 个样本
  - `mixed`: 396 个样本  
  - `random`: 396 个样本
- **原始FJS文件数**: 396 个（每个原始文件产生3个样本）
- **文件格式**: JSON

## 数据转换逻辑

### 转换前（原始数据结构）
```json
{
  "原始文件名": {
    "fjs_path": "路径",
    "features": { ... },
    "performance_data": {
      "initialization_methods": {
        "heuristic": { "性能指标" },
        "mixed": { "性能指标" },
        "random": { "性能指标" }
      }
    }
  }
}
```

### 转换后（新数据结构）
```json
{
  "sample_0001": {
    "sample_id": "sample_0001",
    "original_fjs_path": "原始文件路径",
    "initialization_method": "heuristic",
    "features": { ... },
    "performance_data": {
      "performance_metrics": { "单一方法的性能指标" }
    }
  },
  "sample_0002": {
    "sample_id": "sample_0002", 
    "original_fjs_path": "原始文件路径",
    "initialization_method": "mixed",
    "features": { ... },
    "performance_data": {
      "performance_metrics": { "单一方法的性能指标" }
    }
  }
}
```

## 详细数据结构

### 1. 顶层结构

每个样本包含以下5个主要字段：

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `sample_id` | String | 唯一样本标识符，格式为 "sample_XXXX" |
| `original_fjs_path` | String | 原始FJS文件的相对路径 |
| `initialization_method` | String | 初始化方法（heuristic/mixed/random） |
| `features` | Object | 特征数据对象 |
| `performance_data` | Object | 性能数据对象 |

### 2. Features（特征数据）

特征数据包含4大类特征：

#### 2.1 基础特征 (`basic_features`)

| 特征名 | 类型 | 描述 | 示例值 |
|--------|------|------|--------|
| `num_jobs` | Integer | 作业数量 | 10 |
| `num_machines` | Integer | 机器数量 | 11 |
| `total_operations` | Integer | 总操作数 | 100 |
| `avg_available_machines` | Float | 每个操作平均可用机器数 | 1.1 |
| `std_available_machines` | Float | 可用机器数的标准差 | 0.3 |

#### 2.2 加工时间特征 (`processing_time_features`)

| 特征名 | 类型 | 描述 | 示例值 |
|--------|------|------|--------|
| `processing_time_mean` | Float | 加工时间均值 | 50.93 |
| `processing_time_std` | Float | 加工时间标准差 | 27.19 |
| `processing_time_min` | Float | 最小加工时间 | 2.0 |
| `processing_time_max` | Float | 最大加工时间 | 99.0 |
| `machine_time_variance` | Float | 机器时间方差 | 35.55 |

#### 2.3 析取图特征 (`disjunctive_graphs_features`)

| 特征名 | 类型 | 描述 |
|--------|------|------|
| `nodes_count` | Integer | 图中节点数量 |
| `edges_count` | Integer | 图中边的数量 |
| `initial_labels` | Object | 初始节点标签映射 |
| `solid_labels` | Object | 实线边标签映射（WL算法生成） |
| `dashed_labels` | Object | 虚线边标签映射（WL算法生成） |
| `solid_frequency` | Object | 实线标签频率统计 |
| `dashed_frequency` | Object | 虚线标签频率统计 |

#### 2.4 KDE特征 (`kde_features`)

| 特征名 | 类型 | 描述 |
|--------|------|------|
| `x_grid` | Array | KDE计算的x轴网格点（1000个点，0-999） |
| `density` | Array | 对应每个网格点的概率密度值 |
| `bandwidth` | Float | KDE使用的带宽参数 |

### 3. Performance Data（性能数据）

性能数据记录了算法在该样本上的执行表现：

#### 3.1 元信息

| 字段名 | 类型 | 描述 | 示例值 |
|--------|------|------|--------|
| `meta_heuristic` | String | 使用的元启发式算法 | "HA(GA+TS)" |
| `execution_times` | Integer | 算法执行次数 | 20 |
| `max_iterations` | Integer | 最大迭代次数 | 100 |

#### 3.2 性能指标 (`performance_metrics`)

| 指标名 | 类型 | 描述 |
|--------|------|------|
| `mean` | Float | 多次执行结果的平均makespan |
| `std` | Float | 多次执行结果的标准差 |
| `min` | Float | 多次执行的最小makespan |
| `max` | Float | 多次执行的最大makespan |
| `avg_convergence_generation` | Float | 平均收敛代数 |
| `convergence_generation_std` | Float | 收敛代数的标准差 |

## 样本示例

### 完整样本示例（sample_0001）

```json
{
  "sample_0001": {
    "sample_id": "sample_0001",
    "original_fjs_path": "Barnes/mt10c1.fjs",
    "initialization_method": "heuristic",
    "features": {
      "basic_features": {
        "num_jobs": 10,
        "num_machines": 11,
        "total_operations": 100,
        "avg_available_machines": 1.1,
        "std_available_machines": 0.3
      },
      "processing_time_features": {
        "processing_time_mean": 50.92727272727273,
        "processing_time_std": 27.1908103095376,
        "processing_time_min": 2.0,
        "processing_time_max": 99.0,
        "machine_time_variance": 35.55289256198348
      },
      "disjunctive_graphs_features": {
        "nodes_count": 102,
        "edges_count": 210,
        "initial_labels": {
          "J1O1": "J1O1",
          "J1O2": "J1O2",
          // ... 更多标签
        },
        "solid_labels": {
          "J1O1": "J1O1_H9802",
          // ... WL算法生成的标签
        },
        "dashed_labels": {
          "J1O1": "J1O1_H2003",
          // ... WL算法生成的标签
        },
        "solid_frequency": {
          "J1O1_H9802": 1,
          // ... 标签频率统计
        },
        "dashed_frequency": {
          "J1O1_H2003": 1,
          // ... 标签频率统计
        }
      },
      "kde_features": {
        "x_grid": [0.0, 1.0, 2.0, ..., 999.0],
        "density": [0.079, 0.129, 0.113, ...],
        "bandwidth": 1.1446382709559153
      }
    },
    "performance_data": {
      "meta_heuristic": "HA(GA+TS)",
      "execution_times": 20,
      "max_iterations": 100,
      "performance_metrics": {
        "mean": 1161.05,
        "std": 47.46733087082104,
        "min": 1105,
        "max": 1281,
        "avg_convergence_generation": 16.15,
        "convergence_generation_std": 10.946574806760333
      }
    }
  }
}
```

## 数据集使用说明

### 1. 数据加载

```python
import json

# 加载数据集
with open('converted_fjs_dataset_new.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"总样本数: {len(dataset)}")
```

### 2. 按初始化方法分组

```python
# 按初始化方法分组
method_groups = {}
for sample_id, sample_data in dataset.items():
    method = sample_data['initialization_method']
    if method not in method_groups:
        method_groups[method] = []
    method_groups[method].append(sample_data)

for method, samples in method_groups.items():
    print(f"{method}: {len(samples)} 个样本")
```

### 3. 特征提取

```python
# 提取特定类型的特征
def extract_basic_features(sample_data):
    return sample_data['features']['basic_features']

def extract_performance_metrics(sample_data):
    return sample_data['performance_data']['performance_metrics']

# 示例：获取所有样本的基础特征
basic_features_list = []
for sample_data in dataset.values():
    basic_features_list.append(extract_basic_features(sample_data))
```

## 数据质量保证

### 1. 数据完整性
- 所有样本都包含完整的5个主要字段
- 特征数据保持原始精度，未做修改

### 2. 数据一致性
- 相同原始文件的3个样本（不同初始化方法）具有相同的特征数据
- 样本ID连续递增，从 sample_0001 到 sample_1188
- 每种初始化方法的样本数量完全相等（396个）

### 3. 数据溯源
- `original_fjs_path` 字段记录了样本的原始来源
- 通过该字段可以追溯到原始的FJS问题文件
- 保持了与原始数据集的对应关系

## 技术细节

### 文件大小和性能
- 文件大小：约329 MB
- 加载时间：在普通PC上约需要2-5秒
- 建议使用SSD存储以提高I/O性能

### 内存使用
- 完整加载到内存约需要 800MB - 1GB
- 建议在处理时分批加载以节省内存

### 数据访问模式
- 顺序访问：适合批量处理
- 随机访问：通过 sample_id 快速定位
- 分组访问：按 initialization_method 分组处理

---

## 更新历史

- **2025-08-20**: 初始版本，基于 `analyze_and_convert.py` 输出的数据结构
- 文档版本：1.0
- 数据集版本：converted_fjs_dataset_new.json

---

*此文档由自动化脚本生成，如有疑问请参考源代码 `analyze_and_convert.py`*





