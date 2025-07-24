# 性能对比实验使用说明

## 概述
本目录包含完整的性能对比实验系统，用于比较推荐策略模型与随机初始化方法的性能差异。系统采用**两阶段推荐架构**，通过特征相似度搜索和策略推荐实现最优初始化方法的选择。

## 两阶段推荐系统架构

### 第一阶段：多特征相似度搜索
基于四种特征融合的相似度计算，从历史数据集中找到最相似的问题实例。

#### 特征类型
1. **基础特征** (权重: 30%)
   - 作业数量 (`num_jobs`)
   - 机器数量 (`num_machines`) 
   - 总操作数 (`total_operations`)
   - 平均可用机器数 (`avg_available_machines`)
   - 可用机器数标准差 (`std_available_machines`)

2. **加工时间特征** (权重: 25%)
   - 加工时间均值 (`processing_time_mean`)
   - 加工时间标准差 (`processing_time_std`)
   - 加工时间最小值 (`processing_time_min`)
   - 加工时间最大值 (`processing_time_max`)
   - 机器时间方差 (`machine_time_variance`)

3. **KDE特征** (权重: 20%)
   - 基于核密度估计的加工时间分布特征
   - 使用Jensen-Shannon散度计算相似度

4. **析取图特征** (权重: 25%)
   - 基于Weisfeiler-Lehman算法的图结构特征
   - 捕获问题实例的结构相似性

#### 相似度计算流程
```python
# 核心实现位置：recommend_model_1/initialization_strategy_recommender.py
def stage_one_similarity_search(self, new_data_features, top_k=5):
    # 1. 特征标准化
    normalized_all_features = self.normalize_features(all_features)
    
    # 2. 计算四种特征的相似度
    basic_similarity = self.calculate_euclidean_distance(...)
    processing_similarity = self.calculate_euclidean_distance(...)
    kde_similarity = 1 - self.calculate_js_divergence(...)
    disjunctive_similarity = self.calculate_disjunctive_graph_similarity(...)
    
    # 3. 加权融合
    weighted_similarity = (
        0.3 * basic_similarity +
        0.25 * processing_similarity +
        0.2 * kde_similarity +
        0.25 * disjunctive_similarity
    )
    
    # 4. 返回Top-K最相似样本
    return top_k_candidates
```

### 第二阶段：策略推荐
基于候选样本的性能数据，计算四维性能指标并推荐最优策略。

#### 性能指标
1. **Makespan精度** (权重: 40%)
   - 完工时间性能评估（越小越好）
   - 评分公式：`1.0 / (1.0 + mean_makespan / 1000.0)`

2. **收敛速度** (权重: 25%)
   - 算法收敛效率评估（收敛代数越少越好）
   - 评分公式：`1.0 - (avg_convergence_gen / 100.0)`

3. **稳定性** (权重: 20%)
   - 结果稳定性评估（标准差越小越好）
   - 评分公式：`1.0 / (1.0 + std_makespan / 10.0)`

4. **收敛稳定性** (权重: 15%)
   - 收敛过程稳定性评估（收敛标准差越小越好）
   - 评分公式：`1.0 / (1.0 + convergence_std / 10.0)`

#### 策略推荐流程
```python
# 核心实现位置：recommend_model_1/initialization_strategy_recommender.py
def stage_two_strategy_recommendation(self, candidate_samples, top_k=3):
    # 1. 收集候选样本的策略性能数据
    for fjs_path, similarity_score, _ in candidate_samples:
        # 2. 计算四维性能评分
        performance_score = (
            weights['makespan'] * makespan_score +
            weights['convergence_speed'] * convergence_speed_score +
            weights['stability'] * stability_score +
            weights['convergence_stability'] * convergence_stability_score
        )
        
        # 3. 相似度加权平均
        weighted_avg_score = similarity_weighted_average(performance_score)
    
    # 4. 返回推荐策略
    return recommended_strategies
```

## 文件结构
```
compare_with_random/
├── main_experiment.py              # 主实验文件（统一执行入口）
├── random_initialization_test.py   # 随机初始化性能测试
├── recommended_strategy_test.py    # 推荐策略性能测试
├── performance_comparison.py       # 性能对比分析
├── exp_result/                     # 实验结果目录（自动生成）
└── README.md                       # 本说明文件
```

## 使用方法

### 1. 快速开始
```bash
# 切换到实验目录
cd recommend_model_1/result/compare_with_random

# 运行完整实验（建议使用FJS文件的绝对路径）
python main_experiment.py D:/0-MyCode/Dispatch_Sample_Generator/recommend_model_1/result/new_data_Behnke29.fjs

# 或者使用相对路径（需要确保在正确的目录下运行）
python main_experiment.py ../../result/new_data_Behnke29.fjs
```

### 2. 实验流程
主实验文件会自动执行以下步骤：

1. **随机初始化性能测试**
   - 使用随机初始化方法运行GA算法
   - 执行20次，每次100次迭代
   - 记录Makespan、收敛代数、执行时间

2. **推荐策略性能测试**
   - 调用两阶段推荐系统获取最佳初始化策略
   - 第一阶段：多特征相似度搜索找到候选样本
   - 第二阶段：基于性能指标推荐最优策略
   - 使用推荐策略运行GA算法
   - 执行20次，每次100次迭代
   - 记录性能指标

3. **性能对比分析**
   - 生成性能对比表格
   - 创建对比图表和雷达图
   - 绘制收敛曲线对比
   - 生成详细分析报告

### 3. 输出结果
实验结果保存在 `exp_result/` 目录下：

#### 主要文件
- `random_initialization_results_YYYYMMDD_HHMMSS.json` - 随机初始化测试结果
- `recommended_strategy_results_YYYYMMDD_HHMMSS.json` - 推荐策略测试结果
- `performance_comparison_table_YYYYMMDD_HHMMSS.png` - 性能对比表格
- `performance_comparison_charts.png` - 性能对比图表
- `improvement_radar_chart.png` - 改进率雷达图
- `convergence_comparison.png` - 收敛曲线对比
- `comparison_summary_YYYYMMDD_HHMMSS.md` - 详细对比总结
- `experiment_summary_YYYYMMDD_HHMMSS.md` - 实验总结报告

#### 日志文件
- `main_experiment_YYYYMMDD_HHMMSS.log` - 主实验日志
- `random_initialization_test_YYYYMMDD_HHMMSS.log` - 随机初始化测试日志
- `recommended_strategy_test_YYYYMMDD_HHMMSS.log` - 推荐策略测试日志
- `performance_comparison_YYYYMMDD_HHMMSS.log` - 性能对比分析日志

### 4. 单独运行测试
如果需要单独运行某个测试，可以：

```bash
# 只运行随机初始化测试
python random_initialization_test.py

# 只运行推荐策略测试
python recommended_strategy_test.py

# 只运行性能对比分析
python performance_comparison.py
```

### 5. 自定义参数
可以通过修改脚本中的以下参数来自定义实验：

- `runs = 20` - 执行次数
- `max_iterations = 100` - 最大迭代数
- `fjs_file_path` - FJS文件路径

## 实验指标

### 求解精度 (Makespan)
- 目标：最小化最大完工时间
- 改进率 = (随机初始化均值 - 推荐策略均值) / 随机初始化均值 × 100%

### 收敛效率 (收敛代数)
- 目标：更快收敛到最优解
- 改进率 = (随机初始化收敛代数 - 推荐策略收敛代数) / 随机初始化收敛代数 × 100%

### 时间性能 (执行时间)
- 目标：减少计算时间
- 改进率 = (随机初始化时间 - 推荐策略时间) / 随机初始化时间 × 100%

## 注意事项

1. **依赖文件**：确保以下文件存在：
   - `../labeled_dataset/labeled_fjs_dataset.json` - 标记数据集
   - `../extract_new_data_features.py` - 特征提取模块
   - `../initialization_strategy_recommender.py` - 推荐系统

2. **FJS文件格式**：确保输入的FJS文件格式正确

3. **计算资源**：实验可能需要较长时间，建议在计算资源充足的环境下运行

4. **结果解释**：正改进率表示推荐策略优于随机初始化，负值表示相反

## 故障排除

### 常见问题
1. **文件不存在错误**：检查依赖文件路径是否正确
2. **解析失败**：检查FJS文件格式是否正确
3. **推荐系统错误**：检查标记数据集是否完整

### 日志查看
查看详细的错误信息：
```bash
# 查看主实验日志
cat exp_result/main_experiment_*.log

# 查看具体测试日志
cat exp_result/random_initialization_test_*.log
cat exp_result/recommended_strategy_test_*.log
```

## 扩展功能

### 添加新的测试数据
1. 将新的FJS文件放入 `result/` 目录
2. 运行主实验文件并指定新文件路径
3. 查看生成的对比结果

### 修改实验参数
编辑相应的测试文件，修改 `runs` 和 `max_iterations` 参数

### 自定义分析
修改 `performance_comparison.py` 文件，添加新的分析维度或图表类型 