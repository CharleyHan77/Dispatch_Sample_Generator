# WL算法在 `extract_new_data_features.py` 和 `calculate_weighted_similarity.py` 中的体现分析

## 概述

本文档详细分析了WL（Weisfeiler-Lehman）算法在 `feature_similarity_weighting` 目录下两个核心文件中的具体体现。WL算法作为图结构相似度计算的核心技术，在这两个文件中分别承担了**特征提取**和**特征使用**的不同角色。

---

## WL算法在 `extract_new_data_features.py` 中的体现

### 1. **WL算法函数的导入**
```python
# 导入析取图特征生成相关函数
from comparison_disjunctive_graphs.extract_graph_features import (
    create_disjunctive_graph_with_attributes,
    init_node_labels,
    add_edge_attributes,
    wl_step  # ← 这是WL算法的核心函数
)
```

### 2. **WL算法的完整执行流程**
在 `extract_disjunctive_graph_features` 函数中，WL算法按以下步骤执行：

#### **A. 图构建和初始化**
```python
# 创建析取图
graph = create_disjunctive_graph_with_attributes(parameters, os.path.basename(fjs_path))

# 初始化节点标签
graph = init_node_labels(graph)

# 添加边属性（与compare_graphs_wl.py保持一致）
graph = add_edge_attributes(graph)

# 获取初始标签
initial_labels = {node: graph.nodes[node]['label'] for node in graph.nodes()}
```

#### **B. 双重WL迭代**
```python
# 第一轮WL迭代（实线）
solid_labels = wl_step(graph, 'solid', initial_labels)

# 第二轮WL迭代（虚线）
dashed_labels = wl_step(graph, 'dashed', solid_labels)
```

#### **C. 标签频率统计**
```python
def get_label_frequency(labels):
    """统计标签频率"""
    freq = {}
    for label in labels.values():
        freq[label] = freq.get(label, 0) + 1
    return freq

# 分别获取实线和虚线标签频率
solid_frequency = get_label_frequency(solid_labels)
dashed_frequency = get_label_frequency(dashed_labels)
```

#### **D. 特征存储**
```python
# 构建graph_info，与compare_graphs_wl.py中的graph_info结构完全一致
graph_info = {
    "nodes_count": len(graph.nodes()),
    "edges_count": len(graph.edges()),
    "initial_labels": initial_labels,
    "solid_labels": solid_labels,      # ← 实线WL迭代结果
    "dashed_labels": dashed_labels,    # ← 虚线WL迭代结果
    "solid_frequency": solid_frequency,    # ← 实线标签频率
    "dashed_frequency": dashed_frequency   # ← 虚线标签频率
}
```

---

## WL算法在 `calculate_weighted_similarity.py` 中的体现

### 1. **WL算法函数的导入**
```python
# 导入析取图相关函数
from comparison_disjunctive_graphs.compare_graphs_wl import (
    create_disjunctive_graph_with_attributes,
    init_node_labels,
    add_edge_attributes,
    wl_step  # ← 同样导入WL算法核心函数
)
```

### 2. **WL算法结果的直接使用**
在 `calculate_disjunctive_graph_similarity` 函数中，WL算法的体现主要体现在**使用WL算法生成的特征数据**：

#### **A. 获取WL标签频率**
```python
# 获取实线和虚线标签频率
solid_freq1 = graph_info1['solid_frequency']    # ← WL算法生成的实线标签频率
solid_freq2 = graph_info2['solid_frequency']
dashed_freq1 = graph_info1['dashed_frequency']  # ← WL算法生成的虚线标签频率
dashed_freq2 = graph_info2['dashed_frequency']
```

#### **B. 基于WL标签的相似度计算**
```python
# 计算实线标签的Jaccard相似度
solid_keys1 = set(solid_freq1.keys())
solid_keys2 = set(solid_freq2.keys())
if len(solid_keys1.union(solid_keys2)) > 0:
    solid_jaccard = len(solid_keys1.intersection(solid_keys2)) / len(solid_keys1.union(solid_keys2))
else:
    solid_jaccard = 0.0

# 计算虚线标签的Jaccard相似度
dashed_keys1 = set(dashed_freq1.keys())
dashed_keys2 = set(dashed_freq2.keys())
if len(dashed_keys1.union(dashed_keys2)) > 0:
    dashed_jaccard = len(dashed_keys1.intersection(dashed_keys2)) / len(dashed_keys1.union(dashed_keys2))
else:
    dashed_jaccard = 0.0
```

#### **C. WL标签分布相似度计算**
```python
# 计算标签分布相似度（使用余弦相似度）
all_solid_keys = solid_keys1.union(solid_keys2)
solid_vec1 = np.array([solid_freq1.get(k, 0) for k in all_solid_keys])
solid_vec2 = np.array([solid_freq2.get(k, 0) for k in all_solid_keys])

solid_norm1 = np.linalg.norm(solid_vec1)
solid_norm2 = np.linalg.norm(solid_vec2)

if solid_norm1 == 0 or solid_norm2 == 0:
    solid_cosine = 0.0
else:
    solid_cosine = np.dot(solid_vec1, solid_vec2) / (solid_norm1 * solid_norm2)
```

---

## 两个文件中WL算法体现的对比

### **`extract_new_data_features.py` - WL算法的执行者**
- **直接执行WL算法**：调用 `wl_step` 函数进行双重WL迭代
- **生成WL特征**：创建 `solid_frequency` 和 `dashed_frequency`
- **特征存储**：将WL算法结果保存到JSON文件中

### **`calculate_weighted_similarity.py` - WL算法的使用者**
- **间接使用WL算法**：读取WL算法生成的特征数据
- **相似度计算**：基于WL标签频率计算图结构相似度
- **结果应用**：将WL特征相似度纳入综合加权计算

---

## WL算法的具体工作流程

### 1. **在特征提取阶段（`extract_new_data_features.py`）**
```
FJS文件 → 析取图构建 → 节点标签初始化 → 边属性添加 → 
第一轮WL迭代（实线） → 第二轮WL迭代（虚线） → 
标签频率统计 → 特征存储
```

### 2. **在相似度计算阶段（`calculate_weighted_similarity.py`）**
```
读取WL特征 → 标签集合相似度（Jaccard） → 
标签分布相似度（余弦） → 加权组合 → 
结构相似度融合 → 最终析取图相似度
```

---

## WL算法的核心特点

### 1. **双重WL迭代**
- **实线WL迭代**：捕捉工序顺序约束的结构特征
- **虚线WL迭代**：捕捉机器互斥约束的结构特征
- 这种双重迭代比传统WL算法更能反映FJSP的特殊结构

### 2. **边类型区分**
- 区分实线边（工序顺序）和虚线边（机器互斥）
- 在WL迭代中分别处理不同类型的边
- 更好地保留了FJSP析取图的语义信息

### 3. **标签哈希化**
```python
# 在wl_step函数中体现
combined = f"{current_labels[node]}_{','.join(neighbors)}"
hash_value = hash(combined) % 10000  # 取模确保哈希值在合理范围内
new_labels[node] = f"{current_labels[node]}_H{hash_value}"
```

### 4. **机器信息融合**
```python
# 在虚线WL迭代中附加机器信息
neighbors.append(f"{current_labels[neighbor]}_{machine}")
```

---

## WL标签的具体示例

从JSON数据中可以看到WL标签的格式：
```json
"solid_frequency": {
    "J1O1_H8505": 1,
    "J1O2_H2829": 1,
    "J1O3_H8885": 1,
    // ...
},
"dashed_frequency": {
    "J1O1_H8505_H2038": 1,
    "J1O2_H2829_H9271": 1,
    "J1O3_H8885_H5716": 1,
    // ...
}
```

其中：
- `J1O1_H8505`：表示节点J1O1经过实线WL迭代后的哈希标签
- `J1O1_H8505_H2038`：表示节点J1O1经过虚线WL迭代后的哈希标签

---

## 设计优势

### 1. **特征提取与使用分离**
- **一次计算，多次使用**：WL特征只需提取一次，可以用于多次相似度计算
- **模块化设计**：特征提取和相似度计算分离，便于维护和扩展
- **高效计算**：避免在相似度计算时重复执行WL算法

### 2. **双重WL迭代的优势**
- **更丰富的结构信息**：通过区分边类型，捕捉更细粒度的图结构特征
- **FJSP特定优化**：针对FJSP析取图的特点进行算法优化
- **语义保持**：保留工序顺序和机器互斥的语义信息

### 3. **标签哈希化的优势**
- **计算效率**：将复杂的邻居信息压缩为哈希值，提高计算效率
- **内存优化**：减少标签存储空间
- **数值稳定性**：避免标签空间过大导致的数值问题

---

## 总结

WL算法在这两个文件中的体现体现了**特征提取与特征使用分离**的设计思想：

1. **`extract_new_data_features.py`**：负责执行WL算法，生成图结构特征
2. **`calculate_weighted_similarity.py`**：负责使用WL算法生成的特征，计算相似度

这种设计使得WL算法可以：
- **一次计算，多次使用**：WL特征只需提取一次，可以用于多次相似度计算
- **模块化设计**：特征提取和相似度计算分离，便于维护和扩展
- **高效计算**：避免在相似度计算时重复执行WL算法

WL算法通过双重迭代、边类型区分、标签哈希化等技术，为FJSP析取图的相似度计算提供了强大而高效的支持。 