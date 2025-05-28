import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import json
import pandas as pd
import matplotlib.font_manager as fm
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from feature_extraction.feature_extractor import FeatureExtractor

def create_disjunctive_graph(parameters, instance_name, save_path=None, title="析取图"):
    """
    创建FJSP析取图
    :param parameters: 问题参数
    :param instance_name: 实例名称
    :param save_path: 保存路径
    :param title: 图表标题
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 获取作业数和每作业的操作数
    jobs = parameters['jobs']
    job_count = len(jobs)
    machines_count = parameters['machinesNb']
    max_ops = max(len(job) for job in jobs)
    
    # 计算节点位置
    node_positions = {}
    
    # 为每个作业的工序分配位置
    for job_idx, job in enumerate(jobs):
        # 将作业的所有工序排在一行
        y_pos = job_count - job_idx  # 从上到下排列作业
        for op_idx, op in enumerate(job):
            # 在0到1之间均匀分布工序
            x_pos = op_idx / (len(job) + 1)
            node_name = f'J{job_idx+1}O{op_idx+1}'
            # 获取该操作可用的所有机器
            available_machines = []
            processing_times = []
            for option in op:
                available_machines.append(option[0] + 1)  # 机器编号从1开始
                processing_times.append(option[1])
            G.add_node(node_name, 
                      job=job_idx, 
                      op=op_idx, 
                      available_machines=available_machines,
                      processing_times=processing_times)
            node_positions[node_name] = (x_pos, y_pos)
    
    # 添加虚拟起始节点和终止节点，位于左右两侧居中位置
    node_positions['Start'] = (-0.1, job_count / 2)  # 左侧居中
    node_positions['End'] = (1.1, job_count / 2)     # 右侧居中
    G.add_node('Start')
    G.add_node('End')
    
    # 添加工艺约束（实线）- 只连接同一作业内的相邻工序
    for job_idx, job in enumerate(jobs):
        # 连接起始节点到作业的第一个工序
        first_op = f'J{job_idx+1}O1'
        G.add_edge('Start', first_op, type='conjunctive')
        
        # 连接作业内的相邻工序
        for op_idx in range(len(job)-1):
            current_op = f'J{job_idx+1}O{op_idx+1}'
            next_op = f'J{job_idx+1}O{op_idx+2}'
            G.add_edge(current_op, next_op, type='conjunctive')
        
        # 连接作业的最后一个工序到终止节点
        last_op = f'J{job_idx+1}O{len(job)}'
        G.add_edge(last_op, 'End', type='conjunctive')
    
    # 添加析取约束（虚线）- 为每台机器生成析取弧环
    machine_edges = {}  # 用于存储每台机器的析取弧
    for machine in range(1, machines_count + 1):
        machine_nodes = []  # 存储当前机器可用的所有节点
        
        # 按作业和工序顺序遍历所有节点
        for job_idx, job in enumerate(jobs):
            for op_idx, op in enumerate(job):
                node_name = f'J{job_idx+1}O{op_idx+1}'
                # 检查该工序是否允许在该机器上加工
                if machine in G.nodes[node_name]['available_machines']:
                    machine_nodes.append(node_name)
        
        # 如果找到了至少两个节点，生成析取弧环
        if len(machine_nodes) >= 2:
            # 将节点按顺序连接成环
            edges = []
            for i in range(len(machine_nodes)):
                next_i = (i + 1) % len(machine_nodes)
                current_node = machine_nodes[i]
                next_node = machine_nodes[next_i]
                # 添加边并标记机器信息
                edges.append((current_node, next_node))
            machine_edges[machine] = edges
    
    # 绘制图
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制节点
    nx.draw_networkx_nodes(G, node_positions, 
                          node_color=['red' if node in ['Start', 'End'] else 'lightblue' 
                                    for node in G.nodes()],
                          node_size=1000,
                          alpha=0.7)
    
    # 绘制实线（工艺约束）
    conjunctive_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d.get('type') == 'conjunctive']
    nx.draw_networkx_edges(G, node_positions, 
                          edgelist=conjunctive_edges,
                          edge_color='black',
                          arrows=True,
                          arrowsize=20)
    
    # 绘制析取弧（每个机器一个环）
    edge_labels = {}  # 用于存储边的标签（机器号）
    for machine, edges in machine_edges.items():
        # 为每个机器设置不同的弧度
        base_rad = 0.5  # 基础弧度
        rad = base_rad + (machine % 3) * 0.2  # 在0.5到1.1之间变化
        
        # 绘制该机器的析取弧环
        nx.draw_networkx_edges(G, node_positions,
                             edgelist=edges,
                             edge_color='gray',
                             style='dashed',
                             arrows=False,
                             width=1,
                             alpha=0.6,
                             connectionstyle=f'arc3,rad={rad}')
        
        # 为每条边添加机器号标签
        for edge in edges:
            edge_labels[edge] = f'M{machine}'
    
    # 添加边标签（机器号）
    nx.draw_networkx_edge_labels(G, node_positions,
                               edge_labels=edge_labels,
                               font_size=8)
    
    # 添加节点标签
    labels = {}
    for node in G.nodes():
        if node in ['Start', 'End']:
            labels[node] = node
        else:
            data = G.nodes[node]
            job_idx = data['job']
            op_idx = data['op']
            machines = [f'M{m}' for m in data['available_machines']]
            times = data['processingTime']
            # 格式化标签：作业号、工序号、可用机器和对应加工时间
            machine_info = [f'{m}({t})' for m, t in zip(machines, times)]
            labels[node] = f'J{job_idx+1}O{op_idx+1}\n{", ".join(machine_info)}'
    
    nx.draw_networkx_labels(G, node_positions, labels, font_size=8)
    
    plt.title(f"{title}\n实例: {instance_name}", fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    try:
        # 获取数据集根目录
        dataset_root = os.path.join(project_root, "dataset")
        
        # 遍历数据集目录
        for dataset_name in os.listdir(dataset_root):
            dataset_path = os.path.join(dataset_root, dataset_name)
            
            # 确保是目录
            if not os.path.isdir(dataset_path):
                continue
                
            # 创建对应的输出目录
            output_dir = os.path.join(project_root, "output", "disjunctive_graphs", dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理该数据集下的所有.fjs文件
            for instance_name in os.listdir(dataset_path):
                if not instance_name.endswith('.fjs'):
                    continue
                    
                file_path = os.path.join(dataset_path, instance_name)
                print(f"\n正在处理文件: {file_path}")
                
                try:
                    # 解析数据文件
                    parameters = parser.parse(file_path)
                    
                    # 设置图表标题和保存路径
                    title = f"{dataset_name} {instance_name} 析取图"
                    save_path = os.path.join(output_dir, f"{os.path.splitext(instance_name)[0]}_disjunctive_graph.png")
                    
                    # 创建并保存析取图
                    create_disjunctive_graph(parameters, instance_name, save_path, title)
                    print(f"析取图已保存到: {save_path}")
                    
                except Exception as e:
                    print(f"处理文件 {instance_name} 时发生错误: {str(e)}")
                    continue
        
        print("\n所有数据集处理完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 