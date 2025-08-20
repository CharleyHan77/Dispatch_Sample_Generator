#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比实验主文件
自动执行完整的性能对比实验流程：
1. 随机初始化性能测试
2. 推荐策略性能测试  
3. 性能对比分析
"""

import os
import sys
import time
import datetime
import logging
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

def setup_logging(log_file):
    """设置日志记录"""
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_python_script(script_path, description, logger, fjs_file_path=None, timestamp=None, weights_config=None):
    """
    运行Python脚本
    
    Args:
        script_path: 脚本路径
        description: 脚本描述
        logger: 日志记录器
        fjs_file_path: FJS文件路径
        timestamp: 时间戳
        weights_config: 权重配置文件路径
        
    Returns:
        bool: 是否成功
    """
    logger.info(f"开始执行: {description}")
    logger.info(f"脚本路径: {script_path}")
    
    start_time = time.time()
    
    try:
        # 切换到脚本所在目录
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        # 运行脚本
        cmd = [sys.executable, script_name]
        if fjs_file_path:
            cmd.append(fjs_file_path)
        if timestamp:
            cmd.append(timestamp)
        if weights_config:
            cmd.extend(['--weights-config', weights_config])
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ {description} 执行成功 (耗时: {duration:.2f}秒)")
            if result.stdout:
                logger.info("输出信息:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            return True
        else:
            logger.error(f"✗ {description} 执行失败 (耗时: {duration:.2f}秒)")
            logger.error(f"错误代码: {result.returncode}")
            if result.stderr:
                logger.error("错误信息:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"✗ {description} 执行异常: {e}")
        return False

def check_file_exists(file_path, description, logger):
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
        description: 文件描述
        logger: 日志记录器
        
    Returns:
        bool: 文件是否存在
    """
    if os.path.exists(file_path):
        logger.info(f"✓ {description} 存在: {file_path}")
        return True
    else:
        logger.error(f"✗ {description} 不存在: {file_path}")
        return False

def create_experiment_summary(exp_result_dir, fjs_file, timestamp, logger):
    """
    创建实验总结报告
    
    Args:
        exp_result_dir: 实验结果目录
        fjs_file: FJS文件路径
        timestamp: 时间戳
        logger: 日志记录器
    """
    summary_content = f"""# 性能对比实验总结报告

## 实验概况
- **实验时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **测试数据**: {fjs_file}
- **实验ID**: {timestamp}

## 实验流程
1. ✅ 随机初始化性能测试
2. ✅ 推荐策略性能测试
3. ✅ 性能对比分析

## 生成文件
实验结果保存在: {exp_result_dir}

### 主要输出文件
- `random_initialization_results_{timestamp}.json` - 随机初始化测试结果
- `recommended_strategy_results_{timestamp}.json` - 推荐策略测试结果
- `performance_comparison_table_{timestamp}.png` - 性能对比表格
- `performance_comparison_charts.png` - 性能对比图表
- `improvement_radar_chart.png` - 改进率雷达图
- `convergence_comparison.png` - 收敛曲线对比
- `comparison_summary_{timestamp}.md` - 详细对比总结

## 四维度性能指标
- **Makespan评分**: 完工时间性能评估
- **收敛速度评分**: 算法收敛效率评估
- **稳定性评分**: 结果稳定性评估
- **收敛稳定性评分**: 收敛过程稳定性评估

## 实验完成
所有实验步骤已成功完成，请查看上述文件了解详细结果。
"""
    
    summary_file = os.path.join(exp_result_dir, f'experiment_summary_{timestamp}.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    logger.info(f"实验总结报告已保存: {summary_file}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='性能对比实验主程序 - 支持权重配置')
    parser.add_argument('fjs_file', help='FJS文件路径')
    parser.add_argument('--weights-config', type=str, default=None, help='权重配置文件路径 (JSON格式)')
    
    args = parser.parse_args()
    fjs_file_path = args.fjs_file
    
    # 检查FJS文件是否存在
    if not os.path.exists(fjs_file_path):
        print(f"错误: FJS文件不存在: {fjs_file_path}")
        return
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验结果目录（使用时间戳子目录）
    exp_result_dir = os.path.join("exp_result", f"exp_{timestamp}")
    os.makedirs(exp_result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(exp_result_dir, f"main_experiment_{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("性能对比实验主程序启动")
    logger.info(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"测试数据: {fjs_file_path}")
    logger.info(f"实验结果目录: {exp_result_dir}")
    logger.info(f"实验ID: {timestamp}")
    if args.weights_config:
        logger.info(f"权重配置文件: {args.weights_config}")
    else:
        logger.info("权重配置: 使用默认权重")
    logger.info("=" * 80)
    
    # 记录实验参数
    logger.info("实验参数配置:")
    logger.info(f"  FJS文件: {fjs_file_path}")
    logger.info(f"  执行次数: 20")
    logger.info(f"  最大迭代数: 100")
    logger.info(f"  性能指标: 四维度评分（Makespan、收敛速度、稳定性、收敛稳定性）")
    logger.info("-" * 80)
    
    # 检查必要文件
    logger.info("检查必要文件...")
    
    # 获取项目根目录的绝对路径
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # 依赖文件的绝对路径
    LABELED_DATASET_PATH = os.path.join(BASE_DIR, 'recommend_model_1/labeled_dataset/labeled_fjs_dataset.json')
    FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, 'recommend_model_1/extract_new_data_features.py')
    RECOMMENDER_PATH = os.path.join(BASE_DIR, 'recommend_model_1/initialization_strategy_recommender.py')
    RANDOM_SCRIPT = os.path.join(BASE_DIR, 'recommend_model_1/result/compare_with_random/random_initialization_test.py')
    RECOMMENDED_SCRIPT = os.path.join(BASE_DIR, 'recommend_model_1/result/compare_with_random/recommended_strategy_test.py')
    COMPARISON_SCRIPT = os.path.join(BASE_DIR, 'recommend_model_1/result/compare_with_random/performance_comparison.py')

    # 检查标记数据集
    if not check_file_exists(LABELED_DATASET_PATH, "标记数据集", logger):
        logger.error("缺少标记数据集，实验终止")
        return
    # 检查特征提取模块
    if not check_file_exists(FEATURE_EXTRACTOR_PATH, "特征提取模块", logger):
        logger.error("缺少特征提取模块，实验终止")
        return
    # 检查推荐系统
    if not check_file_exists(RECOMMENDER_PATH, "推荐系统", logger):
        logger.error("缺少推荐系统，实验终止")
        return
    
    logger.info("所有必要文件检查完成")
    logger.info("-" * 80)
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 步骤1: 随机初始化性能测试
    logger.info("步骤1: 随机初始化性能测试")
    random_script = RANDOM_SCRIPT
    
    # 直接通过参数调用子脚本，传递时间戳
    if not run_python_script(random_script, "随机初始化性能测试", logger, os.path.abspath(fjs_file_path), timestamp):
        logger.error("随机初始化测试失败，实验终止")
        return
    
    # 检查随机初始化结果
    random_result_files = list(Path(exp_result_dir).glob("random_initialization_results_*.json"))
    if not random_result_files:
        logger.error("未找到随机初始化测试结果，实验终止")
        return
    
    logger.info("步骤1完成")
    logger.info("-" * 80)
    
    # 步骤2: 推荐策略性能测试
    logger.info("步骤2: 推荐策略性能测试")
    recommended_script = RECOMMENDED_SCRIPT
    
    # 直接通过参数调用子脚本，传递时间戳和权重配置
    if not run_python_script(recommended_script, "推荐策略性能测试", logger, os.path.abspath(fjs_file_path), timestamp, args.weights_config):
        logger.error("推荐策略测试失败，实验终止")
        return
    
    # 检查推荐策略结果
    recommended_result_files = list(Path(exp_result_dir).glob("recommended_strategy_results_*.json"))
    if not recommended_result_files:
        logger.error("未找到推荐策略测试结果，实验终止")
        return
    
    logger.info("步骤2完成")
    logger.info("-" * 80)
    
    # 步骤3: 性能对比分析
    logger.info("步骤3: 性能对比分析")
    comparison_script = COMPARISON_SCRIPT
    
    # 传递时间戳给对比分析脚本
    if not run_python_script(comparison_script, "性能对比分析", logger, None, timestamp):
        logger.error("性能对比分析失败，实验终止")
        return
    
    logger.info("步骤3完成")
    logger.info("-" * 80)
    
    # 记录总结束时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 创建实验总结
    create_experiment_summary(exp_result_dir, fjs_file_path, timestamp, logger)
    
    # 记录最终统计信息
    logger.info("=" * 80)
    logger.info("性能对比实验完成")
    logger.info(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {total_duration:.2f}秒")
    logger.info(f"实验结果目录: {exp_result_dir}")
    logger.info("=" * 80)
    
    # 列出生成的文件
    logger.info("生成的文件列表:")
    for file_path in Path(exp_result_dir).glob("*"):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            logger.info(f"  {file_path.name} ({file_size} bytes)")
    
    logger.info("=" * 80)
    logger.info("实验成功完成！")

if __name__ == "__main__":
    main() 