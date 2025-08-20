# 权重配置实验指南

## 概述

本指南提供了完整的权重配置方案，用于实验不同权重设置对推荐系统性能的影响。包含特征权重和性能指标评价权重的配置。

## 配置文件说明

### 1. 权重配置文件结构

```json
{
  "config_name": "配置名称",
  "description": "配置描述",
  "weights": {
    "feature_weights": {
      "basic_features": { ... },           // 基础特征权重
      "processing_time_features": { ... }, // 加工时间特征权重
      "kde_similarity_weight": 0.2,        // KDE特征权重
      "disjunctive_similarity_weight": 0.25 // 析取图特征权重
    },
    "performance_weights": {
      "makespan_weight": 0.4,              // Makespan权重
      "convergence_speed_weight": 0.25,    // 收敛速度权重
      "stability_weight": 0.2,             // 稳定性权重
      "convergence_stability_weight": 0.15 // 收敛稳定性权重
    },
    "recommendation_weights": {
      "feature_weight": 0.6,               // 特征相似度权重
      "performance_weight": 0.4            // 性能指标权重
    }
  }
}
```

### 2. 预设配置文件

| 配置文件 | 特点 | 适用场景 |
|---------|------|----------|
| `weights_config_template.json` | 默认平衡配置 | 一般性实验 |
| `weights_scale_focused.json` | 强调规模特征 | 规模敏感的问题 |
| `weights_time_focused.json` | 强调时间特征 | 时间敏感的调度 |
| `weights_performance_focused.json` | 强调性能指标 | 性能优先的场景 |
| `weights_balanced.json` | 完全均衡配置 | 基准对比实验 |

## 使用方法

### 1. 单个配置测试

```bash
# 使用特定权重配置运行实验
python main_experiment.py new_Behnke3.fjs --weights-config weights_scale_focused.json
```

### 2. 批量配置测试

```bash
# 使用实验脚本测试所有预设配置
python test_weight_configurations.py new_Behnke3.fjs

# 测试指定的配置文件
python test_weight_configurations.py new_Behnke3.fjs --configs weights_scale_focused.json weights_time_focused.json
```

### 3. 自定义配置

1. 复制 `weights_config_template.json`
2. 修改权重值
3. 保存为新的配置文件
4. 运行实验测试

## 权重调优指南

### 特征权重调优

#### 基础特征 (总权重建议: 25-35%)
- **num_jobs** (5-12%): 作业数量，影响问题规模
- **num_machines** (5-12%): 机器数量，影响资源约束
- **total_operations** (3-8%): 总操作数，反映复杂度
- **avg_available_machines** (2-8%): 平均可用机器，影响灵活性
- **std_available_machines** (1-5%): 机器可用性变异，反映约束复杂性

#### 加工时间特征 (总权重建议: 20-30%)
- **processing_time_mean** (6-12%): 平均加工时间，最重要的时间特征
- **processing_time_std** (4-10%): 时间标准差，反映时间分布
- **processing_time_min** (2-6%): 最小时间，影响下界
- **processing_time_max** (2-6%): 最大时间，影响上界
- **machine_time_variance** (1-5%): 机器时间差异

#### 其他特征权重
- **kde_similarity_weight** (15-25%): KDE分布特征
- **disjunctive_similarity_weight** (20-30%): 图结构特征

### 性能权重调优

#### 性能指标权重 (总和必须为1.0)
- **makespan_weight** (30-50%): 完工时间，核心优化目标
- **convergence_speed_weight** (15-35%): 收敛速度，效率指标
- **stability_weight** (10-30%): 稳定性，可靠性指标
- **convergence_stability_weight** (5-20%): 收敛稳定性

#### 推荐权重平衡
- **feature_weight** (30-70%): 特征相似度重要性
- **performance_weight** (30-70%): 性能历史重要性

## 实验建议

### 1. 系统性实验设计

```bash
# 步骤1: 基准测试
python main_experiment.py test_data.fjs --weights-config weights_config_template.json

# 步骤2: 特征导向测试
python main_experiment.py test_data.fjs --weights-config weights_scale_focused.json
python main_experiment.py test_data.fjs --weights-config weights_time_focused.json

# 步骤3: 性能导向测试
python main_experiment.py test_data.fjs --weights-config weights_performance_focused.json

# 步骤4: 均衡配置测试
python main_experiment.py test_data.fjs --weights-config weights_balanced.json
```

### 2. 敏感性分析

对单个权重进行微调，观察性能变化：

```python
# 示例：调整num_jobs权重从0.05到0.15，步长0.02
for weight in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]:
    # 创建配置文件
    # 运行实验
    # 记录结果
```

### 3. 交叉验证

使用不同的测试数据验证权重配置的泛化能力：

```bash
# 多个数据集测试
for data in ["new_Behnke3.fjs", "new_Behnke29.fjs", "other_data.fjs"]; do
    python main_experiment.py $data --weights-config weights_optimal.json
done
```

## 结果分析

### 1. 性能指标比较

关注以下指标的改进：
- **Makespan改进率**: (随机-推荐)/随机 × 100%
- **收敛速度改进**: 收敛代数减少程度
- **稳定性提升**: 标准差降低程度

### 2. 权重有效性评估

- **特征权重影响**: 观察不同特征权重对推荐结果的影响
- **性能权重影响**: 分析性能权重对策略选择的影响
- **整体平衡**: 评估特征相似度与性能历史的平衡效果

### 3. 最优配置识别

基于多次实验结果，识别：
- 特定问题类型的最优权重
- 通用性较好的权重配置
- 特殊场景的专用权重

## 注意事项

1. **权重约束**:
   - 各类特征权重总和应合理
   - 性能权重必须总和为1.0
   - 单个权重建议在0.01-0.5之间

2. **实验控制**:
   - 使用相同的随机种子确保可重现性
   - 控制实验环境变量
   - 记录详细的实验参数

3. **统计显著性**:
   - 运行多次实验取平均值
   - 进行统计检验验证改进的显著性
   - 考虑置信区间

4. **计算资源**:
   - 批量实验需要较长时间
   - 建议在计算资源充足时运行
   - 可以调整运行次数和迭代数

## 故障排除

### 常见问题

1. **配置文件格式错误**:
   - 检查JSON格式是否正确
   - 验证权重值是否为数值类型

2. **权重值不合理**:
   - 检查权重总和是否合理
   - 确认权重值在有效范围内

3. **实验运行失败**:
   - 查看错误日志
   - 检查依赖文件是否存在
   - 验证FJS文件格式

### 调试建议

```bash
# 验证配置文件格式
python -c "import json; print(json.load(open('config.json')))"

# 检查权重总和
python -c "
import json
config = json.load(open('config.json'))
basic_sum = sum(config['weights']['feature_weights']['basic_features'].values())
processing_sum = sum(config['weights']['feature_weights']['processing_time_features'].values())
print(f'基础特征权重总和: {basic_sum}')
print(f'加工时间特征权重总和: {processing_sum}')
"
```

## 扩展应用

1. **多目标优化**: 调整权重进行多目标优化实验
2. **问题分类**: 针对不同问题类型设计专用权重
3. **在线学习**: 基于历史性能动态调整权重
4. **集成方法**: 结合多个权重配置的结果
