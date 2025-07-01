import networkx as nx
import numpy as np
import os
import sys
import json
from collections import Counter
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import glob

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.test_disjunctive_graph_correct import create_disjunctive_graph
from initial_validation.utils import parser

def init_node_labels(graph: nx.DiGraph) -> nx.DiGraph:
    """
    初始化节点标签 - 摘要化版本
    :param graph: 析取图
    :return: 带有标签的图
    """
    for node in graph.nodes():
        if node == 'Start':
            graph.nodes[node]['label'] = 'START'
        elif node == 'End':
            graph.nodes[node]['label'] = 'END'
        else:
            # 实际节点：只用工序ID，不带机器和时间信息
            data = graph.nodes[node]
            job = data['job']
            op = data['operation']
            job_op = f"J{job}O{op}"
            graph.nodes[node]['label'] = job_op
    
    return graph

def create_disjunctive_graph_with_attributes(parameters: Dict[str, Any], instance_name: str) -> nx.DiGraph:
    """
    创建带有完整边属性的析取图
    :param parameters: 从parser.parse()得到的解析结果
    :param instance_name: 实例名称
    :return: 析取图的DiGraph对象
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 获取作业数和机器数
    jobs = parameters['jobs']
    job_count = len(jobs)
    machines_count = parameters['machinesNb']
    
    # 为每个作业的工序分配位置
    for job_idx, job in enumerate(jobs):
        for op_idx, op in enumerate(job):
            # 在0到1之间均匀分布工序
            x_pos = (op_idx + 1) / (len(job) + 1.5)  # 增加间距
            node_name = f'J{job_idx+1}O{op_idx+1}'
            
            # 获取该操作可用的所有机器和加工时间
            available_machines = []
            processing_times = []
            for option in op:
                available_machines.append(option['machine'])  # 机器编号从1开始
                processing_times.append(option['processingTime'])
            
            # 添加节点
            G.add_node(node_name, 
                      job=job_idx+1, 
                      operation=op_idx+1, 
                      available_machines=available_machines,
                      processing_times=processing_times)
    
    # 添加虚拟起始节点和终止节点
    G.add_node('Start', type='virtual')
    G.add_node('End', type='virtual')
    
    # 添加工艺约束（实线）
    for job_idx, job in enumerate(jobs):
        # 连接起始节点到作业的第一个工序
        first_op = f'J{job_idx+1}O1'
        G.add_edge('Start', first_op, type='conjunctive', edge_type='solid')
        
        # 连接作业内的相邻工序
        for op_idx in range(len(job)-1):
            current_op = f'J{job_idx+1}O{op_idx+1}'
            next_op = f'J{job_idx+1}O{op_idx+2}'
            G.add_edge(current_op, next_op, type='conjunctive', edge_type='solid')
        
        # 连接最后一个工序到终止节点
        last_op = f'J{job_idx+1}O{len(job)}'
        G.add_edge(last_op, 'End', type='conjunctive', edge_type='solid')
    
    # 为每个机器生成析取弧环
    machine_edges = {}  # 用于存储每台机器的析取弧
    
    # 对每台机器单独处理
    for machine in range(1, machines_count + 1):
        machine_nodes = []  # 存储当前机器可用的所有节点（按遍历顺序）
        
        # 按作业和工序顺序遍历所有节点
        for job_idx, job in enumerate(jobs):
            for op_idx, op in enumerate(job):
                node_name = f'J{job_idx+1}O{op_idx+1}'
                # 检查该工序是否允许在该机器上加工
                if machine in G.nodes[node_name]['available_machines']:
                    machine_nodes.append(node_name)
        
        # 如果找到了至少两个节点，生成析取弧环
        if len(machine_nodes) >= 2:
            edges = []
            # 将相邻节点连接，并首尾相连形成环
            for i in range(len(machine_nodes)):
                current_node = machine_nodes[i]
                next_node = machine_nodes[(i + 1) % len(machine_nodes)]
                edges.append((current_node, next_node))
            machine_edges[machine] = edges
    
    # 添加析取弧（虚线边）
    for machine, edges in machine_edges.items():
        for u, v in edges:
            G.add_edge(u, v, type='disjunctive', edge_type='dashed', machine=f'M{machine}')
    
    return G

def add_edge_attributes(graph: nx.DiGraph) -> nx.DiGraph:
    """
    为边添加类型属性
    :param graph: 析取图
    :return: 带有边属性的图
    """
    # 实线边：工序顺序约束
    for u, v, data in graph.edges(data=True):
        if data.get('type') == 'conjunctive':
            data['edge_type'] = 'solid'
        else:
            # 虚线边：机器互斥约束
            data['edge_type'] = 'dashed'
            # 从节点数据中获取机器信息
            if u in graph.nodes and 'available_machines' in graph.nodes[u]:
                # 这里简化处理，实际应该根据边的具体机器分配来确定
                data['machine'] = 'M1'  # 简化处理
    
    return graph

def wl_step(graph: nx.DiGraph, edge_type: str, current_labels: Dict) -> Dict:
    """
    执行一步WL迭代 - 带哈希化
    :param graph: 析取图
    :param edge_type: 边类型 ('solid' 或 'dashed')
    :param current_labels: 当前节点标签
    :return: 更新后的节点标签
    """
    new_labels = {}
    
    for node in graph.nodes():
        # 收集邻居信息
        neighbors = []
        
        if edge_type == 'solid':
            # 实线WL迭代：仅通过实线边聚合邻居标签
            for neighbor in graph.predecessors(node):
                if graph.edges[neighbor, node].get('edge_type') == 'solid':
                    neighbors.append(current_labels[neighbor])
            for neighbor in graph.successors(node):
                if graph.edges[node, neighbor].get('edge_type') == 'solid':
                    neighbors.append(current_labels[neighbor])
        else:
            # 虚线WL迭代：仅通过虚线边聚合邻居标签，并附加机器ID
            for neighbor in graph.predecessors(node):
                if graph.edges[neighbor, node].get('edge_type') == 'dashed':
                    machine = graph.edges[neighbor, node].get('machine', 'M1')
                    neighbors.append(f"{current_labels[neighbor]}_{machine}")
            for neighbor in graph.successors(node):
                if graph.edges[node, neighbor].get('edge_type') == 'dashed':
                    machine = graph.edges[node, neighbor].get('machine', 'M1')
                    neighbors.append(f"{current_labels[neighbor]}_{machine}")
        
        # 排序邻居标签以确保一致性
        neighbors.sort()
        
        # 生成新标签：当前标签 + 邻居标签的哈希值（减少唯一性）
        combined = f"{current_labels[node]}_{','.join(neighbors)}"
        # 使用简单的哈希函数，将长标签映射到较短的标识符
        hash_value = hash(combined) % 10000  # 取模确保哈希值在合理范围内
        new_labels[node] = f"{current_labels[node]}_H{hash_value}"
    
    return new_labels

def extract_graph_features(fjs_path: str) -> Dict:
    """提取单个FJS文件的图标签特征"""
    print(f"处理文件: {fjs_path}")
    
    try:
        # 解析FJS文件
        parameters = parser.parse(fjs_path)
        
        # 创建析取图
        graph = create_disjunctive_graph_with_attributes(parameters, os.path.basename(fjs_path))
        
        # 初始化节点标签
        graph = init_node_labels(graph)
        
        # 添加边属性（与compare_graphs_wl.py保持一致）
        graph = add_edge_attributes(graph)
        
        # 获取初始标签
        initial_labels = {node: graph.nodes[node]['label'] for node in graph.nodes()}
        
        # 第一轮WL迭代（实线）
        solid_labels = wl_step(graph, 'solid', initial_labels)
        
        # 第二轮WL迭代（虚线）
        dashed_labels = wl_step(graph, 'dashed', solid_labels)
        
        # 分别统计实线和虚线WL的标签频率，然后加权
        def get_label_frequency(labels):
            """统计标签频率"""
            freq = {}
            for label in labels.values():
                freq[label] = freq.get(label, 0) + 1
            return freq
        
        # 分别获取实线和虚线标签频率
        solid_frequency = get_label_frequency(solid_labels)
        dashed_frequency = get_label_frequency(dashed_labels)
        
        # 构建graph_info，与compare_graphs_wl.py中的graph_info结构完全一致
        graph_info = {
            "nodes_count": len(graph.nodes()),
            "edges_count": len(graph.edges()),
            "initial_labels": initial_labels,
            "solid_labels": solid_labels,
            "dashed_labels": dashed_labels,
            "solid_frequency": solid_frequency,
            "dashed_frequency": dashed_frequency
        }
        
        return graph_info
        
    except Exception as e:
        print(f"  处理文件 {fjs_path} 时发生错误: {str(e)}")
        return {"error": str(e)}

def main():
    """主函数：在现有dataset_features.json中添加析取图特征"""
    print("开始在现有dataset_features.json中添加析取图特征...")
    
    # 读取现有的dataset_features.json文件
    dataset_features_path = os.path.join(project_root, "output", "dataset_features.json")
    
    if not os.path.exists(dataset_features_path):
        print(f"错误：找不到文件 {dataset_features_path}")
        return
    
    print(f"读取现有特征文件: {dataset_features_path}")
    
    with open(dataset_features_path, 'r', encoding='utf-8') as f:
        dataset_features = json.load(f)
    
    print(f"现有文件包含 {len(dataset_features)} 个FJS实例")
    
    # 统计信息
    total_processed = 0
    total_successful = 0
    total_failed = 0
    
    # 为每个FJS文件添加析取图特征
    for fjs_key in dataset_features.keys():
        total_processed += 1
        
        # 构建完整的FJS文件路径
        fjs_path = os.path.join(project_root, "dataset", fjs_key)
        
        if not os.path.exists(fjs_path):
            print(f"警告：找不到FJS文件 {fjs_path}")
            dataset_features[fjs_key]["disjunctive_graphs_features"] = {"error": "FJS file not found"}
            total_failed += 1
            continue
        
        # 提取图特征
        graph_features = extract_graph_features(fjs_path)
        
        # 添加到现有特征中
        dataset_features[fjs_key]["disjunctive_graphs_features"] = graph_features
        
        if "error" not in graph_features:
            total_successful += 1
            print(f"  ✓ 成功添加图特征")
        else:
            total_failed += 1
            print(f"  ✗ 添加图特征失败")
    
    # 保存更新后的文件
    print(f"\n保存更新后的特征文件...")
    with open(dataset_features_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_features, f, indent=2, ensure_ascii=False)
    
    print(f"更新后的特征文件已保存到: {dataset_features_path}")
    
    # 输出统计信息
    print("\n处理统计:")
    print(f"  总文件数: {total_processed}")
    print(f"  成功处理: {total_successful}")
    print(f"  处理失败: {total_failed}")
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main() 