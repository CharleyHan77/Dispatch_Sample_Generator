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




## 4.参考repo
- 基于框架repo：https://github.com/CharleyHan77/Flexible-Job-Shop-Scheduling-Problem.git
- 涉及到的fjsp数据集：https://github.com/Lei-Kun/FJSP-benchmarks.git
- 验证使用的初步元启发算法：D:\0-科研资料\2-code-Solver\FJSP_GA [1]
- 相关调度规则：https://github.com/Lei-Kun/Dispatching-rules-for-FJSP.git [2]


## 5.参考文献
- [1] Xinyu Li and Liang Gao. An effective hybrid genetic algorithm and tabu search for flexible job shop scheduling problem.International Journal of ProductionEconomics, 174 :93 – 110, 2016
- [2] Lei K, Guo P, Zhao W, et al. A multi-action deep reinforcement learning framework for flexible Job-shop scheduling problem[J]. Expert Systems with Applications, 2022, 205: 117796.

