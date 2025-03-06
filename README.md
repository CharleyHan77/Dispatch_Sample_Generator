# Dispatch-Sample-Generator

## 1.项目描述
- 2026 毕设
- 期望通过fjsp场景下的历史`生产数据`，生成推荐模型所需的对应`调度样本数据`。

## 2.使用说明

### 2.1 安装包

`pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2.2 执行
根目录下
`python main.py`

## 3.目录结构
- app 主要逻辑代码
- dataset fjsp生产数据集
- result 运行结果（一般为导出的excel，用于构成调度样本）
- 

## 4.参考repo
- 基于框架repo：https://github.com/CharleyHan77/Flexible-Job-Shop-Scheduling-Problem.git
- 涉及到的fjsp数据集：https://github.com/Lei-Kun/FJSP-benchmarks.git
- 相关调度规则：https://github.com/Lei-Kun/Dispatching-rules-for-FJSP.git