#!/usr/bin/env python

# This module creates a population of random OS and MS chromosomes

import random

from initial_validation import config


#from src import config


def generateOS(parameters):
    jobs = parameters['jobs']

    OS = []
    i = 0
    for job in jobs:
        for op in job:
            # 每个作业的每个工序
            # 初始化OS：每个作业号依次插入，数量为工序数
            OS.append(i)
        i = i + 1
    # 随机打乱生成
    random.shuffle(OS)

    return OS


def generateMS(parameters):
    jobs = parameters['jobs']

    MS = []
    for job in jobs:
        for op in job:
            # MS中每个值代表的是当前作业的当前工序的可选机器集中的索引下标，不是机器下标
            randomMachine = random.randint(0, len(op)-1)
            MS.append(randomMachine)

    return MS


def initializePopulation_random(parameters):
    gen1 = []
    # 根据设置的种群数量中每个个体生成OS、MS基因序列
    # 选择不同的初始化种群初始方法
    for i in range(config.popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    return gen1

##############################



def generateOS_heuristic(parameters):
    jobs = parameters['jobs']
    OS = []
    # 确保 jobs[i] 是有效的
    # for idx, job in enumerate(jobs):
    #     if isinstance(job, list):  # 如果是列表
    #         for op in job:
    #             if isinstance(op, tuple) and len(op) == 2:  # 确保每个工序是元组并且有两个元素
    #                 pass
    #             else:
    #                 print(f"Invalid operation format in job {idx}: {op}")
    #     else:
    #         print(f"Invalid job format at index {idx}: {job}")

    # 按照每个作业的加工时间总和对作业排序
    job_priority = sorted(range(len(jobs)),
                          key=lambda i: sum(op[1] for op in jobs[i] if isinstance(op, tuple) and len(op) == 2),
                          reverse=True)

    # 打印每个作业的结构，帮助调试
    # for idx, job in enumerate(jobs):
    #     print(f"Job {idx}: {job}")

    # 按照作业优先级生成工序
    for job_idx in job_priority:
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)
    return OS

def generateMS_heuristic(parameters):
    jobs = parameters['jobs']
    MS = []

    for job in jobs:
        for op in job:
            # 获取当前工序的所有机器及其加工时间
            machines = [(m['machine'], m['processingTime']) for m in op]

            # 选择加工时间最短的机器的索引
            min_time_machine = min(range(len(machines)), key=lambda m: machines[m][1])
            MS.append(min_time_machine)

    return MS

def initializePopulation_heuristic(parameters):
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_heuristic(parameters)
        MS = generateMS_heuristic(parameters)
        gen1.append((OS, MS))
    return gen1

#######################################################################

def generateOS_mixed(parameters):
    """
    generateOS_mixed：按作业的总加工时间排序。
    """
    jobs = parameters['jobs']
    OS = []

    # 确保 jobs[i] 是有效的
    # for idx, job in enumerate(jobs):
    #     if isinstance(job, list):  # 如果是列表
    #         for op in job:
    #             if isinstance(op, tuple) and len(op) == 2:  # 确保每个工序是元组并且有两个元素
    #                 pass
    #             else:
    #                 print(f"Invalid operation format in job {idx}: {op}")
    #     else:
    #         print(f"Invalid job format at index {idx}: {job}")

    # 按照每个作业的加工时间总和对作业排序
    job_priority = sorted(range(len(jobs)),
                          key=lambda i: sum(op[1] for op in jobs[i] if isinstance(op, tuple) and len(op) == 2),
                          reverse=True)

    # 打印每个作业的结构，帮助调试
    # for idx, job in enumerate(jobs):
    #     print(f"Job {idx}: {job}")

    # 按照作业优先级生成工序
    for job_idx in job_priority:
        for op in jobs[job_idx]:
            if op:  # 确保工序有效
                OS.append(job_idx)

    # 随机打乱生成，以增加多样性
    #random.shuffle(OS)
    return OS

def generateMS_mixed(parameters):
    """
    随机选择机器，这样能够增加多样性和探索空间。
    :param parameters:
    :return:
    """
    jobs = parameters['jobs']
    MS = []
    for job in jobs:
        for op in job:
            # 随机选择机器，增加变异性
            randomMachine = random.randint(0, len(op) - 1)
            MS.append(randomMachine)
    return MS

def initializePopulation_mixed(parameters):
    gen1 = []
    for i in range(config.popSize):
        OS = generateOS_mixed(parameters)
        MS = generateMS_mixed(parameters)
        gen1.append((OS, MS))
    return gen1
