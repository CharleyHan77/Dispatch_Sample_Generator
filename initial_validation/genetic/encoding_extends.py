#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
扩展的初始化方法模块
包含八种初始化策略：FIFO_SPT、FIFO_EET、MOPNR_SPT、MOPNR_EET、LWKR_SPT、LWKR_EET、MWKR_SPT、MWKR_EET
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any

from initial_validation import config


def generateOS_FIFO(parameters: Dict[str, Any]) -> List[int]:
    """
    FIFO (First In First Out) 工序排序：选择队列中最早到达的作业
    按照作业编号顺序生成工序，模拟先进先出的调度策略
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 工序序列
    """
    jobs = parameters['jobs']
    OS = []
    
    # FIFO策略：按照作业编号顺序生成工序
    # 每个作业的所有工序连续排列，模拟先进先出的调度
    for job_idx in range(len(jobs)):
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)
    
    return OS


def generateMS_SPT(parameters: Dict[str, Any]) -> List[int]:
    """
    SPT (Shortest Processing Time) 机器选择：选择具有最短加工时间的机器来执行作业的操作
    为每个工序选择加工时间最短的机器
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 机器选择序列
    """
    jobs = parameters['jobs']
    MS = []
    
    # SPT策略：为每个工序选择加工时间最短的机器
    for job in jobs:
        for op in job:
            if isinstance(op, list) and len(op) > 0:
                # 获取当前工序的所有机器及其加工时间
                machine_times = [(machine_idx, machine['processingTime']) 
                               for machine_idx, machine in enumerate(op)]
                
                # 选择加工时间最短的机器的索引
                min_time_machine = min(machine_times, key=lambda x: x[1])
                MS.append(min_time_machine[0])
            else:
                # 如果没有可用机器，选择第一个
                MS.append(0)
    
    return MS


def generateMS_EET(parameters: Dict[str, Any]) -> List[int]:
    """
    EET (Earliest End Time) 机器选择：选择最早处于空闲状态的机器
    为每个工序选择最早结束时间的机器，考虑机器负载
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 机器选择序列
    """
    jobs = parameters['jobs']
    MS = []
    
    # EET策略：为每个工序选择最早结束时间的机器
    # 初始化机器负载（所有机器初始负载为0）
    machine_loads = [0] * parameters['machinesNb']
    
    for job in jobs:
        for op in job:
            if isinstance(op, list) and len(op) > 0:
                # 计算每个机器的结束时间（当前负载 + 加工时间）
                machine_end_times = []
                for machine_idx, machine in enumerate(op):
                    machine_id = machine['machine']
                    processing_time = machine['processingTime']
                    end_time = machine_loads[machine_id] + processing_time
                    machine_end_times.append((machine_idx, end_time))
                
                # 选择最早结束时间的机器
                earliest_end_machine = min(machine_end_times, key=lambda x: x[1])
                selected_machine_idx = earliest_end_machine[0]
                selected_machine_id = op[selected_machine_idx]['machine']
                
                # 更新机器负载
                machine_loads[selected_machine_id] = earliest_end_machine[1]
                
                MS.append(selected_machine_idx)
            else:
                # 如果没有可用机器，选择第一个
                MS.append(0)
    
    return MS


def generateOS_MOPNR(parameters: Dict[str, Any]) -> List[int]:
    """
    MOPNR (Most Operations Not Ready) 工序排序：选择剩余待完成操作数最多的作业
    按照作业的工序数量排序，工序数量多的作业优先调度
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 工序序列
    """
    jobs = parameters['jobs']
    OS = []
    
    # MOPNR策略：按照作业的工序数量排序（工序数量多的优先）
    # 计算每个作业的工序数量
    job_operation_counts = [(job_idx, len(jobs[job_idx])) for job_idx in range(len(jobs))]
    
    # 按照工序数量降序排序（工序数量多的优先）
    job_priority = sorted(job_operation_counts, key=lambda x: x[1], reverse=True)
    
    # 按照作业优先级生成工序
    for job_idx, _ in job_priority:
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)
    
    return OS


def generateOS_LWKR(parameters: Dict[str, Any]) -> List[int]:
    """
    LWKR (Least Work Remaining) 工序排序：选择剩余总处理时间最少的作业
    按照作业的总处理时间排序，处理时间少的作业优先调度
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 工序序列
    """
    jobs = parameters['jobs']
    OS = []
    
    # LWKR策略：计算每个作业的总处理时间
    job_workload = []
    for job_idx, job in enumerate(jobs):
        total_processing_time = 0
        for op in job:
            if isinstance(op, list) and len(op) > 0:
                # 计算该工序在所有机器上的平均处理时间
                op_times = [machine['processingTime'] for machine in op]
                total_processing_time += sum(op_times) / len(op_times)
        job_workload.append((job_idx, total_processing_time))
    
    # 按照总处理时间升序排序（处理时间少的优先）
    job_priority = sorted(job_workload, key=lambda x: x[1])
    
    # 按照作业优先级生成工序
    for job_idx, _ in job_priority:
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)
    
    return OS


def generateOS_MWKR(parameters: Dict[str, Any]) -> List[int]:
    """
    MWKR (Most Work Remaining) 工序排序：选择剩余总处理时间最多的作业
    按照作业的总处理时间排序，处理时间多的作业优先调度
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 工序序列
    """
    jobs = parameters['jobs']
    OS = []
    
    # MWKR策略：计算每个作业的总处理时间
    job_workload = []
    for job_idx, job in enumerate(jobs):
        total_processing_time = 0
        for op in job:
            if isinstance(op, list) and len(op) > 0:
                # 计算该工序在所有机器上的平均处理时间
                op_times = [machine['processingTime'] for machine in op]
                total_processing_time += sum(op_times) / len(op_times)
        job_workload.append((job_idx, total_processing_time))
    
    # 按照总处理时间降序排序（处理时间多的优先）
    job_priority = sorted(job_workload, key=lambda x: x[1], reverse=True)
    
    # 按照作业优先级生成工序
    for job_idx, _ in job_priority:
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)
    
    return OS


def generateMS_LoadBalanced(parameters: Dict[str, Any]) -> List[int]:
    """
    负载均衡机器选择：考虑机器负载进行选择
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 机器选择序列
    """
    jobs = parameters['jobs']
    MS = []
    
    # TODO: 实现负载均衡机器选择逻辑
    # 考虑机器当前负载和加工时间
    for job in jobs:
        for op in job:
            # 获取当前工序的所有机器及其加工时间
            machines = [(m['machine'], m['processingTime']) for m in op]
            
            # 随机选择机器（后续可扩展为负载均衡策略）
            random_machine = random.randint(0, len(machines) - 1)
            MS.append(random_machine)
    
    return MS


def generateMS_Adaptive(parameters: Dict[str, Any]) -> List[int]:
    """
    自适应机器选择：根据工序特征自适应选择机器
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[int]: 机器选择序列
    """
    jobs = parameters['jobs']
    MS = []
    
    # TODO: 实现自适应机器选择逻辑
    # 根据工序特征（如加工时间、机器数量等）自适应选择
    for job in jobs:
        for op in job:
            # 获取当前工序的所有机器及其加工时间
            machines = [(m['machine'], m['processingTime']) for m in op]
            
            # 根据工序特征选择机器（当前使用SPT策略）
            min_time_machine = min(range(len(machines)), key=lambda m: machines[m][1])
            MS.append(min_time_machine)
    
    return MS


# ============================================================================
# 八种初始化方法的完整实现
# ============================================================================

def initializePopulation_FIFO_SPT(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    FIFO + SPT 初始化方法
    工序排序：FIFO (First In First Out)
    机器选择：SPT (Shortest Processing Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_FIFO(parameters)
        MS = generateMS_SPT(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_FIFO_EET(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    FIFO + EET 初始化方法
    工序排序：FIFO (First In First Out)
    机器选择：EET (Earliest End Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_FIFO(parameters)
        MS = generateMS_EET(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_MOPNR_SPT(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    MOPNR + SPT 初始化方法
    工序排序：MOPNR (Most Operations Not Ready)
    机器选择：SPT (Shortest Processing Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_MOPNR(parameters)
        MS = generateMS_SPT(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_MOPNR_EET(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    MOPNR + EET 初始化方法
    工序排序：MOPNR (Most Operations Not Ready)
    机器选择：EET (Earliest End Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_MOPNR(parameters)
        MS = generateMS_EET(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_LWKR_SPT(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    LWKR + SPT 初始化方法
    工序排序：LWKR (Least Work Remaining)
    机器选择：SPT (Shortest Processing Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_LWKR(parameters)
        MS = generateMS_SPT(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_LWKR_EET(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    LWKR + EET 初始化方法
    工序排序：LWKR (Least Work Remaining)
    机器选择：EET (Earliest End Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_LWKR(parameters)
        MS = generateMS_EET(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_MWKR_SPT(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    MWKR + SPT 初始化方法
    工序排序：MWKR (Most Work Remaining)
    机器选择：SPT (Shortest Processing Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_MWKR(parameters)
        MS = generateMS_SPT(parameters)
        gen1.append((OS, MS))
    return gen1


def initializePopulation_MWKR_EET(parameters: Dict[str, Any]) -> List[Tuple[List[int], List[int]]]:
    """
    MWKR + EET 初始化方法
    工序排序：MWKR (Most Work Remaining)
    机器选择：EET (Earliest End Time)
    
    Args:
        parameters: 问题参数字典
        
    Returns:
        List[Tuple[List[int], List[int]]]: 种群列表，每个个体为(OS, MS)元组
    """
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_MWKR(parameters)
        MS = generateMS_EET(parameters)
        gen1.append((OS, MS))
    return gen1


# ============================================================================
# 辅助函数
# ============================================================================

def get_init_method_info() -> Dict[str, Dict[str, str]]:
    """
    获取所有初始化方法的信息
    
    Returns:
        Dict[str, Dict[str, str]]: 初始化方法信息字典
    """
    return {
        "FIFO_SPT": {
            "name": "FIFO + SPT",
            "description": "先进先出 + 最短加工时间",
            "os_strategy": "FIFO",
            "ms_strategy": "SPT",
            "os_detail": "选择队列中最早到达的作业",
            "ms_detail": "选择具有最短加工时间的机器"
        },
        "FIFO_EET": {
            "name": "FIFO + EET",
            "description": "先进先出 + 最早结束时间",
            "os_strategy": "FIFO",
            "ms_strategy": "EET",
            "os_detail": "选择队列中最早到达的作业",
            "ms_detail": "选择最早处于空闲状态的机器"
        },
        "MOPNR_SPT": {
            "name": "MOPNR + SPT",
            "description": "剩余操作数最多 + 最短加工时间",
            "os_strategy": "MOPNR",
            "ms_strategy": "SPT",
            "os_detail": "选择剩余待完成操作数最多的作业",
            "ms_detail": "选择具有最短加工时间的机器"
        },
        "MOPNR_EET": {
            "name": "MOPNR + EET",
            "description": "剩余操作数最多 + 最早结束时间",
            "os_strategy": "MOPNR",
            "ms_strategy": "EET",
            "os_detail": "选择剩余待完成操作数最多的作业",
            "ms_detail": "选择最早处于空闲状态的机器"
        },
        "LWKR_SPT": {
            "name": "LWKR + SPT",
            "description": "剩余工作量最少 + 最短加工时间",
            "os_strategy": "LWKR",
            "ms_strategy": "SPT",
            "os_detail": "选择剩余总处理时间最少的作业",
            "ms_detail": "选择具有最短加工时间的机器"
        },
        "LWKR_EET": {
            "name": "LWKR + EET",
            "description": "剩余工作量最少 + 最早结束时间",
            "os_strategy": "LWKR",
            "ms_strategy": "EET",
            "os_detail": "选择剩余总处理时间最少的作业",
            "ms_detail": "选择最早处于空闲状态的机器"
        },
        "MWKR_SPT": {
            "name": "MWKR + SPT",
            "description": "剩余工作量最多 + 最短加工时间",
            "os_strategy": "MWKR",
            "ms_strategy": "SPT",
            "os_detail": "选择剩余总处理时间最多的作业",
            "ms_detail": "选择具有最短加工时间的机器"
        },
        "MWKR_EET": {
            "name": "MWKR + EET",
            "description": "剩余工作量最多 + 最早结束时间",
            "os_strategy": "MWKR",
            "ms_strategy": "EET",
            "os_detail": "选择剩余总处理时间最多的作业",
            "ms_detail": "选择最早处于空闲状态的机器"
        }
    }


def validate_init_method(init_method: str) -> bool:
    """
    验证初始化方法是否有效
    
    Args:
        init_method: 初始化方法名称
        
    Returns:
        bool: 是否有效
    """
    valid_methods = [
        "FIFO_SPT", "FIFO_EET", "MOPNR_SPT", "MOPNR_EET",
        "LWKR_SPT", "LWKR_EET", "MWKR_SPT", "MWKR_EET"
    ]
    return init_method in valid_methods


def get_init_method_function(init_method: str):
    """
    根据初始化方法名称获取对应的函数
    
    Args:
        init_method: 初始化方法名称
        
    Returns:
        function: 对应的初始化函数
    """
    method_map = {
        "FIFO_SPT": initializePopulation_FIFO_SPT,
        "FIFO_EET": initializePopulation_FIFO_EET,
        "MOPNR_SPT": initializePopulation_MOPNR_SPT,
        "MOPNR_EET": initializePopulation_MOPNR_EET,
        "LWKR_SPT": initializePopulation_LWKR_SPT,
        "LWKR_EET": initializePopulation_LWKR_EET,
        "MWKR_SPT": initializePopulation_MWKR_SPT,
        "MWKR_EET": initializePopulation_MWKR_EET
    }
    
    if init_method not in method_map:
        raise ValueError(f"不支持的初始化方法: {init_method}")
    
    return method_map[init_method]


if __name__ == "__main__":
    # 测试代码
    print("初始化方法信息:")
    info = get_init_method_info()
    for method, details in info.items():
        print(f"{method}: {details['name']} - {details['description']}")
    
    print(f"\n支持的初始化方法数量: {len(info)}")
