# 初始化策略推荐系统使用说明

## 概述

初始化策略推荐系统是一个基于历史数据的智能推荐工具，能够为新的FJSP（柔性作业车间调度问题）实例推荐最优的初始化策略。

## 功能特点

- **两阶段推荐流程**：先进行相似度检索，再进行策略推荐
- **多特征融合**：结合基础特征、加工时间特征、KDE特征和析取图特征
- **命令行接口**：支持灵活的参数配置
- **统一输出管理**：自动创建时间戳子目录，便于结果管理
- **可视化结果**：生成相似度对比图和策略推荐图

## 使用方法

### 基本用法

```bash
python initialization_strategy_recommender.py <fjs_file>
```

### 完整参数

```bash
python initialization_strategy_recommender.py <fjs_file> [选项]
```

### 参数说明

- `fjs_file`：输入的FJS文件路径（必需）
- `--top-k-similar`：阶段一返回的最相似样本数量（默认：5）
- `--top-k-strategies`：阶段二推荐的策略数量（默认：3）
- `--output-dir`：输出目录（默认：result/recommender_output）
- `--help`：显示帮助信息

### 使用示例

1. **基本推荐**：
   ```bash
   python initialization_strategy_recommender.py new_data.fjs
   ```

2. **自定义参数**：
   ```bash
   python initialization_strategy_recommender.py new_data.fjs --top-k-similar 3 --top-k-strategies 2
   ```

3. **自定义输出目录**：
   ```bash
   python initialization_strategy_recommender.py new_data.fjs --output-dir my_results
   ```

## 输出结构

每次运行会在输出目录下创建一个以文件名和时间戳命名的子目录：

```
output_dir/
└── filename_YYYYMMDD_HHMMSS/
    ├── recommendation_log.log          # 详细日志
    ├── recommendation_results.json     # 推荐结果JSON
    └── visualization/                  # 可视化图表
        ├── similarity_comparison.png   # 相似度对比图
        └── strategy_recommendation.png # 策略推荐图
```

## 推荐结果说明

### 阶段一：相似度检索
系统会计算新数据与历史数据的多维度相似度：
- **基础特征相似度**：作业数、机器数、总加工时间等
- **加工时间特征相似度**：加工时间分布特征
- **KDE相似度**：核密度估计特征
- **析取图相似度**：析取图结构特征
- **综合加权相似度**：以上特征的加权组合

### 阶段二：策略推荐
基于最相似的历史样本，推荐最优的初始化策略：
- **random**：随机初始化
- **heuristic**：启发式初始化
- **mixed**：混合初始化

## 注意事项

1. 确保输入的FJS文件格式正确
2. 系统需要预先训练好的标记数据集（labeled_fjs_dataset.json）
3. 推荐结果基于历史数据，仅供参考
4. 每次运行都会生成独立的输出目录，便于结果对比

## 错误处理

- 如果输入文件不存在，系统会提示错误并退出
- 如果特征提取失败，会在日志中记录详细信息
- 所有异常都会被捕获并记录到日志文件中 