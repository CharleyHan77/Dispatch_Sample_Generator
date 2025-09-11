#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试转换脚本 - 简化版本
"""

import json
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """设置日志记录"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'test_convert_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("开始测试")
    
    try:
        # 设置路径
        current_dir = Path(__file__).parent
        json_file = current_dir / 'labeled_fjs_dataset.json'
        
        logger.info(f"当前目录: {current_dir}")
        logger.info(f"JSON文件: {json_file}")
        logger.info(f"文件存在: {json_file.exists()}")
        
        if not json_file.exists():
            logger.error("文件不存在")
            return
            
        # 测试加载数据
        logger.info("开始加载JSON数据...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载 {len(data)} 个实例")
        
        # 检查第一个实例的结构
        if data:
            first_key = list(data.keys())[0]
            first_instance = data[first_key]
            logger.info(f"第一个实例: {first_key}")
            logger.info(f"实例结构: {list(first_instance.keys())}")
            
            if 'features' in first_instance:
                features = first_instance['features']
                logger.info(f"特征结构: {list(features.keys())}")
                
            if 'performance_data' in first_instance:
                perf_data = first_instance['performance_data']
                logger.info(f"性能数据结构: {list(perf_data.keys())}")
                
                if 'initialization_methods' in perf_data:
                    init_methods = perf_data['initialization_methods']
                    logger.info(f"初始化方法: {list(init_methods.keys())}")
                    
                    for method, method_data in init_methods.items():
                        logger.info(f"{method}: {list(method_data.keys())}")
        
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()


