# Dispatch-Sample-Generator
【更新】

## 1.项目描述
2026 毕设
- 1.验证基于不同数据验证初始化对元启发求解影响（求解精度、收敛效率）
- 2.通过fjsp场景下的历史`生产数据`，生成推荐模型所需的对应`调度样本数据`
    - 2.1 对每组历史数据提取数据类型的数据特征
    - 2.2 对每组历史数据构建fjsp业务场景的析取图
    - `调度样本数据`，json期望包含：数据集、组数据、数据特征（基本特征/加工时间特征 + 析取图）、元启发算法、执行次数、最大迭代次数、每种初始化方案（mean、std、min、max、平均收敛代数、收敛代数标准差）。【待扩展】
- 3.基于dataset中的数据结构，模拟生成新的生产数据并同样提取特征

## 2.使用说明

### 2.1 安装包

`pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2.2 执行/可视化
xxx

## 3.目录结构
- app 参考引用的元启发式算法框架中相关函数封装
- dataset 目前收集到的fjsp生产数据集，作为历史数据集使用
- feature_extraction 对dataset每组历史数据生成10维特征数据（基础特征、加工时间特征）
- initial_validation 对历史数据 基于不同数据验证初始化对元启发求解影响
    - initial_validation\ga_fjsp.py 参考的元启发式算法框架
    - initial_validation\config.py 元启发算法框架相关参数
    - initial_validation\new_init_population.py 使用ga_fjsp.py中的元启发式算法模拟不同的初始化方案
    - initial_validation\feature_graph_analysis.py 对10维特征数据进行的特征相关性描述图（不能很好体现业务场景）
    - initial_validation\disjunctive_graph.py 生成fjsp析取图
    - initial_validation\output_format.py 格式化输出json（验证统计结果 + 10维特征数据 + 析取图）
- output 对历史数据 记录结果输出 json/png
    - output\convergence_curves 不同初始化方案下的收敛曲线
    - output\disjunctive_graphs 每组历史数据析取图
    - output\feature_graphs 10维特征数据得到的特征相关性描述图
    - output\record_result 每组历史数据的基于不同初始化方案下的统计结果与特征（可用于推荐模型的输入：`调度样本数据`）
- new_data 用于构建一组新的fjsp数据
    - new_data\generate_fjsp_data.py 生成一个新的fjs文件，数据结构与dataset一致
    - new_data\generate_disjunctive_graph.py 基于生成的new_data\new_fjsp_data.fjs使用历史数据相同方法生成析取图
    - new_data\extract_features.py 基于new_data生成数据值特征
- comparison_graph_structure 比较新数据与历史数据的析取图结构，用于实现特征相似度对比
- comparison_disjunctive_graphs 基于析取图计算GED,用于比较图结构之间相似性

- test_ 开头文件/目录为测试需要

## 3.目录结构-新

### 核心功能模块

#### 3.1 特征提取模块 (`feature_extraction/`)
- **`dataset_feature_extractor.py`** - 历史数据集特征提取器，提取基本特征和加工时间特征
- **`feature_extractor.py`** - 通用特征提取工具，支持单文件特征提取
- **`dataset_features.json`** - 历史数据集特征存储文件

#### 3.2 KDE概率密度估计模块 (`PDF_KDE_generator/`)
- **`generate_kde.py`** - KDE参数生成器，计算带宽、标准差、IQR等统计参数
- **`draw_pdf.py`** - KDE概率密度图绘制工具
- **`base_config.py`** - KDE基础配置文件，存储带宽、x_grid等参数

#### 3.3 特征相似度权重计算模块 (`feature_similarity_weighting/`)
- **`extract_new_data_features.py`** - 新数据特征提取器，结合基本特征和KDE生成
- **`calculate_weighted_similarity.py`** - 加权相似度计算器，支持多维度特征比较
- **`similarity_analysis_report.md`** - 相似度分析报告文档
- **新数据子目录** - 包含不同前缀的新数据文件（new_data_f/, new_data_j/, new_data_m/, new_data_o/, new_data_pt/, new_data_ptr/）

#### 3.4 FJS数据生成模块 (`fjs_generator/`)
- **`fjs_generator.py`** - FJSP数据生成器，创建符合历史数据格式的新实例

#### 3.5 新数据处理模块 (`new_data/`)
- **`generate_fjsp_data.py`** - 生成新的FJSP数据文件
- **`generate_disjunctive_graph.py`** - 基于新数据生成析取图
- **`extract_features.py`** - 提取新数据特征
- **`new_fjsp_data.fjs`** - 生成的新FJSP数据文件
- **`new_fjsp_features.json`** - 新数据特征文件

### 输出结果目录 (`output/`)
- **`PDF_KDE_generator/`** - KDE分析结果和参数文件
- **`dataset_features.json`** - 历史数据集特征汇总
- **`validation_results.json`** - 验证结果数据
- **`graph_comparisons/`** - 图结构比较结果
- **`disjunctive_graphs/`** - 析取图可视化结果
- **`feature_graphs/`** - 特征相关性图表
- **`convergence_curves/`** - 算法收敛曲线
- **`record_result/`** - 调度样本数据记录

### 验证和比较模块
- **`initial_validation/`** - 初始验证模块，包含GA算法和初始化方案验证
- **`comparison_graph_structure/`** - 图结构比较分析
- **`comparison_disjunctive_graphs/`** - 析取图相似度计算（GED）
- **`validation/`** - 验证相关工具和脚本

### 数据集和测试
- **`dataset/`** - 历史FJSP数据集
- **`test_dataset/`** - 测试数据集
- **`test_output/`** - 测试输出结果
- **`logs/`** - 日志文件目录

### 核心配置文件
- **`base_config.py`** - 项目基础配置文件，包含KDE参数和全局设置
- **`Schema.py`** - 数据模式定义
- **`parse_fjs.py`** - FJS文件解析工具
- **`FIFO_SPT.py`** - 调度规则实现
- **`test.py`** - 测试脚本

### 应用框架
- **`app/`** - 元启发式算法框架封装

### 主要功能流程
1. **历史数据处理**：`dataset/` → `feature_extraction/` → `PDF_KDE_generator/` → `output/`
2. **新数据生成**：`fjs_generator/` → `new_data/` → `feature_similarity_weighting/`
3. **相似度分析**：`feature_similarity_weighting/calculate_weighted_similarity.py` → 相似度报告
4. **验证分析**：`initial_validation/` → `output/validation_results.json`

## 4.参考repo
- 基于框架repo：https://github.com/CharleyHan77/Flexible-Job-Shop-Scheduling-Problem.git
- 涉及到的fjsp数据集：https://github.com/Lei-Kun/FJSP-benchmarks.git
- 验证使用的初步元启发算法：D:\0-科研资料\2-code-Solver\FJSP_GA [1]
- 相关调度规则：https://github.com/Lei-Kun/Dispatching-rules-for-FJSP.git [2]


## 5.参考文献
- [1] Xinyu Li and Liang Gao. An effective hybrid genetic algorithm and tabu search for flexible job shop scheduling problem.International Journal of ProductionEconomics, 174 :93 – 110, 2016
- [2] Lei K, Guo P, Zhao W, et al. A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem[J]. Expert Systems with Applications, 2022, 205: 117796.

