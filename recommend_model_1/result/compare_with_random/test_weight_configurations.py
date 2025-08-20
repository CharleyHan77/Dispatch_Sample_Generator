#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重配置实验脚本
自动测试不同权重配置对推荐系统性能的影响
"""

import os
import sys
import json
import time
import datetime
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """设置日志记录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"weight_experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_weight_config(config_file):
    """加载权重配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载权重配置文件失败: {e}")
        return None

def run_experiment(fjs_file, config_file, logger):
    """运行单个权重配置实验"""
    config = load_weight_config(config_file)
    if not config:
        return None
    
    config_name = config.get('config_name', os.path.basename(config_file))
    logger.info(f"开始实验: {config_name}")
    logger.info(f"配置文件: {config_file}")
    
    # 运行主实验
    start_time = time.time()
    try:
        cmd = [sys.executable, "main_experiment.py", fjs_file, "--weights-config", config_file]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"实验完成: {config_name} (耗时: {duration:.2f}秒)")
            return {
                'config_name': config_name,
                'config_file': config_file,
                'duration': duration,
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            logger.error(f"实验失败: {config_name}")
            logger.error(f"错误信息: {result.stderr}")
            return {
                'config_name': config_name,
                'config_file': config_file,
                'duration': duration,
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except Exception as e:
        logger.error(f"实验异常: {config_name} - {e}")
        return None

def extract_performance_metrics(experiment_result):
    """从实验结果中提取性能指标"""
    if not experiment_result or not experiment_result['success']:
        return None
    
    # 这里需要根据实际的输出格式来解析性能指标
    # 暂时返回基本信息
    return {
        'config_name': experiment_result['config_name'],
        'duration': experiment_result['duration'],
        'success': experiment_result['success']
    }

def compare_configurations(results, logger):
    """比较不同配置的性能"""
    logger.info("=" * 80)
    logger.info("权重配置性能比较")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if r and r['success']]
    
    if not successful_results:
        logger.warning("没有成功的实验结果可供比较")
        return
    
    logger.info(f"成功完成的实验: {len(successful_results)}/{len(results)}")
    logger.info("-" * 80)
    
    # 按执行时间排序
    successful_results.sort(key=lambda x: x['duration'])
    
    for i, result in enumerate(successful_results, 1):
        logger.info(f"{i}. {result['config_name']}")
        logger.info(f"   执行时间: {result['duration']:.2f}秒")
        logger.info(f"   配置文件: {result['config_file']}")
        logger.info("-" * 40)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='权重配置实验脚本')
    parser.add_argument('fjs_file', help='FJS文件路径')
    parser.add_argument('--configs', nargs='+', 
                       default=['weights_config_template.json', 'weights_scale_focused.json', 
                               'weights_time_focused.json', 'weights_performance_focused.json', 'weights_balanced.json'],
                       help='权重配置文件列表')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("权重配置实验开始")
    logger.info(f"测试文件: {args.fjs_file}")
    logger.info(f"配置文件数量: {len(args.configs)}")
    logger.info("=" * 80)
    
    # 检查FJS文件是否存在
    if not os.path.exists(args.fjs_file):
        logger.error(f"FJS文件不存在: {args.fjs_file}")
        return
    
    # 检查配置文件
    valid_configs = []
    for config_file in args.configs:
        if os.path.exists(config_file):
            valid_configs.append(config_file)
            logger.info(f"✓ 配置文件存在: {config_file}")
        else:
            logger.warning(f"⚠ 配置文件不存在: {config_file}")
    
    if not valid_configs:
        logger.error("没有有效的配置文件")
        return
    
    # 运行实验
    results = []
    total_start_time = time.time()
    
    for i, config_file in enumerate(valid_configs, 1):
        logger.info(f"运行实验 {i}/{len(valid_configs)}: {config_file}")
        result = run_experiment(args.fjs_file, config_file, logger)
        results.append(result)
        
        # 短暂休息，避免系统负载过高
        if i < len(valid_configs):
            logger.info("等待5秒后开始下一个实验...")
            time.sleep(5)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 比较结果
    compare_configurations(results, logger)
    
    logger.info("=" * 80)
    logger.info("权重配置实验完成")
    logger.info(f"总耗时: {total_duration:.2f}秒")
    logger.info(f"成功实验: {sum(1 for r in results if r and r['success'])}/{len(results)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
