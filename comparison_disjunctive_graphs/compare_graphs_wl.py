import networkx as nx
import numpy as np
import os
import sys
import json
from collections import Counter
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

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

def dual_wl_similarity(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """
    计算两个图的双重WL相似度
    :param graph1: 第一个析取图
    :param graph2: 第二个析取图
    :return: 相似度值 (0-1)
    """
    # 初始化节点标签
    graph1 = init_node_labels(graph1)
    graph2 = init_node_labels(graph2)
    
    # 添加边属性
    graph1 = add_edge_attributes(graph1)
    graph2 = add_edge_attributes(graph2)
    
    # 初始标签
    labels1 = {node: graph1.nodes[node]['label'] for node in graph1.nodes()}
    labels2 = {node: graph2.nodes[node]['label'] for node in graph2.nodes()}
    
    # 第一轮WL迭代（实线）
    labels_solid1 = wl_step(graph1, 'solid', labels1)
    labels_solid2 = wl_step(graph2, 'solid', labels2)
    
    # 第二轮WL迭代（虚线）
    labels_dashed1 = wl_step(graph1, 'dashed', labels_solid1)  # 输入实线迭代后的标签
    labels_dashed2 = wl_step(graph2, 'dashed', labels_solid2)
    
    # 分别统计实线和虚线WL的标签频率，然后加权
    def get_label_frequency(labels):
        """统计标签频率"""
        freq = {}
        for label in labels.values():
            freq[label] = freq.get(label, 0) + 1
        return freq
    
    # 分别获取实线和虚线标签频率
    solid_freq1 = get_label_frequency(labels_solid1)
    solid_freq2 = get_label_frequency(labels_solid2)
    dashed_freq1 = get_label_frequency(labels_dashed1)
    dashed_freq2 = get_label_frequency(labels_dashed2)
    
    # 计算实线相似度
    all_solid_keys = set(solid_freq1.keys()).union(set(solid_freq2.keys()))
    solid_vec1 = np.array([solid_freq1.get(k, 0) for k in all_solid_keys])
    solid_vec2 = np.array([solid_freq2.get(k, 0) for k in all_solid_keys])
    
    solid_norm1 = np.linalg.norm(solid_vec1)
    solid_norm2 = np.linalg.norm(solid_vec2)
    
    if solid_norm1 == 0 or solid_norm2 == 0:
        solid_similarity = 0.0
    else:
        solid_similarity = np.dot(solid_vec1, solid_vec2) / (solid_norm1 * solid_norm2)
    
    # 计算虚线相似度
    all_dashed_keys = set(dashed_freq1.keys()).union(set(dashed_freq2.keys()))
    dashed_vec1 = np.array([dashed_freq1.get(k, 0) for k in all_dashed_keys])
    dashed_vec2 = np.array([dashed_freq2.get(k, 0) for k in all_dashed_keys])
    
    dashed_norm1 = np.linalg.norm(dashed_vec1)
    dashed_norm2 = np.linalg.norm(dashed_vec2)
    
    if dashed_norm1 == 0 or dashed_norm2 == 0:
        dashed_similarity = 0.0
    else:
        dashed_similarity = np.dot(dashed_vec1, dashed_vec2) / (dashed_norm1 * dashed_norm2)
    
    # 加权组合相似度（实线权重0.6，虚线权重0.4）
    weighted_similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
    
    return weighted_similarity

def compare_graphs_wl(graph1_name: str, graph2_name: str, graph1: nx.DiGraph, graph2: nx.DiGraph, output_dir: str) -> Dict:
    """
    比较两个图的WL相似度并保存详细结果
    :param graph1_name: 第一个图名称
    :param graph2_name: 第二个图名称
    :param graph1: 第一个析取图
    :param graph2: 第二个析取图
    :param output_dir: 输出目录
    :return: 比较结果字典
    """
    print(f"开始比较 {graph1_name} 和 {graph2_name}...")
    
    # 初始化节点标签
    graph1 = init_node_labels(graph1)
    graph2 = init_node_labels(graph2)
    
    # 添加边属性
    graph1 = add_edge_attributes(graph1)
    graph2 = add_edge_attributes(graph2)
    
    # 初始标签
    labels1 = {node: graph1.nodes[node]['label'] for node in graph1.nodes()}
    labels2 = {node: graph2.nodes[node]['label'] for node in graph2.nodes()}
    
    # 第一轮WL迭代（实线）
    print("  执行实线WL迭代...")
    labels_solid1 = wl_step(graph1, 'solid', labels1)
    labels_solid2 = wl_step(graph2, 'solid', labels2)
    
    # 第二轮WL迭代（虚线）
    print("  执行虚线WL迭代...")
    labels_dashed1 = wl_step(graph1, 'dashed', labels_solid1)
    labels_dashed2 = wl_step(graph2, 'dashed', labels_solid2)
    
    # 分别统计实线和虚线WL的标签频率，然后加权
    def get_label_frequency(labels):
        """统计标签频率"""
        freq = {}
        for label in labels.values():
            freq[label] = freq.get(label, 0) + 1
        return freq
    
    # 分别获取实线和虚线标签频率
    solid_freq1 = get_label_frequency(labels_solid1)
    solid_freq2 = get_label_frequency(labels_solid2)
    dashed_freq1 = get_label_frequency(labels_dashed1)
    dashed_freq2 = get_label_frequency(labels_dashed2)
    
    # 计算实线相似度
    all_solid_keys = set(solid_freq1.keys()).union(set(solid_freq2.keys()))
    solid_vec1 = np.array([solid_freq1.get(k, 0) for k in all_solid_keys])
    solid_vec2 = np.array([solid_freq2.get(k, 0) for k in all_solid_keys])
    
    solid_norm1 = np.linalg.norm(solid_vec1)
    solid_norm2 = np.linalg.norm(solid_vec2)
    
    if solid_norm1 == 0 or solid_norm2 == 0:
        solid_similarity = 0.0
    else:
        solid_similarity = np.dot(solid_vec1, solid_vec2) / (solid_norm1 * solid_norm2)
    
    # 计算虚线相似度
    all_dashed_keys = set(dashed_freq1.keys()).union(set(dashed_freq2.keys()))
    dashed_vec1 = np.array([dashed_freq1.get(k, 0) for k in all_dashed_keys])
    dashed_vec2 = np.array([dashed_freq2.get(k, 0) for k in all_dashed_keys])
    
    dashed_norm1 = np.linalg.norm(dashed_vec1)
    dashed_norm2 = np.linalg.norm(dashed_vec2)
    
    if dashed_norm1 == 0 or dashed_norm2 == 0:
        dashed_similarity = 0.0
    else:
        dashed_similarity = np.dot(dashed_vec1, dashed_vec2) / (dashed_norm1 * dashed_norm2)
    
    # 加权组合相似度（实线权重0.6，虚线权重0.4）
    similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
    
    # 构建结果字典
    result = {
        "graph1_name": graph1_name,
        "graph2_name": graph2_name,
        "similarity": similarity,
        "graph1_info": {
            "nodes_count": len(graph1.nodes()),
            "edges_count": len(graph1.edges()),
            "initial_labels": labels1,
            "solid_labels": labels_solid1,
            "dashed_labels": labels_dashed1,
            "solid_frequency": solid_freq1,
            "dashed_frequency": dashed_freq1
        },
        "graph2_info": {
            "nodes_count": len(graph2.nodes()),
            "edges_count": len(graph2.edges()),
            "initial_labels": labels2,
            "solid_labels": labels_solid2,
            "dashed_labels": labels_dashed2,
            "solid_frequency": solid_freq2,
            "dashed_frequency": dashed_freq2
        },
        "comparison_info": {
            "solid_similarity": solid_similarity,
            "dashed_similarity": dashed_similarity,
            "weighted_similarity": similarity,
            "solid_keys_count": len(all_solid_keys),
            "dashed_keys_count": len(all_dashed_keys),
            "solid_common_keys": len(set(solid_freq1.keys()).intersection(set(solid_freq2.keys()))),
            "dashed_common_keys": len(set(dashed_freq1.keys()).intersection(set(dashed_freq2.keys())))
        }
    }
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, f"{graph1_name}_vs_{graph2_name}_wl_comparison.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  相似度: {similarity:.4f}")
    print(f"  结果已保存到: {output_file}")
    
    return result

def main():
    """主函数：比较Brandimarte数据集中的Mk01.fjs与其他文件的相似度"""
    print("开始基于双重WL的析取图相似度对比...")
    
    # 设置路径
    dataset_root = os.path.join(project_root, "dataset", "Brandimarte")
    output_dir = os.path.join(project_root, "output", "test_compare_wl")
    os.makedirs(output_dir, exist_ok=True)
    
    # 基准文件
    base_file = "Mk01.fjs"
    base_path = os.path.join(dataset_root, base_file)
    
    if not os.path.exists(base_path):
        print(f"错误：找不到基准文件 {base_path}")
        return
    
    # 解析基准文件并创建析取图
    print(f"解析基准文件: {base_file}")
    base_parameters = parser.parse(base_path)
    base_graph = create_disjunctive_graph_with_attributes(base_parameters, base_file)
    
    # 获取所有其他.fjs文件
    other_files = [f for f in os.listdir(dataset_root) if f.endswith('.fjs') and f != base_file]
    other_files.sort()  # 按文件名排序
    
    print(f"找到 {len(other_files)} 个文件用于比较")
    
    # 存储所有比较结果
    all_results = []
    
    # 逐个比较
    for other_file in other_files:
        print(f"\n处理文件: {other_file}")
        other_path = os.path.join(dataset_root, other_file)
        
        try:
            # 解析文件并创建析取图
            other_parameters = parser.parse(other_path)
            other_graph = create_disjunctive_graph_with_attributes(other_parameters, other_file)
            
            # 执行WL相似度比较
            result = compare_graphs_wl(
                os.path.splitext(base_file)[0], 
                os.path.splitext(other_file)[0],
                base_graph, 
                other_graph, 
                output_dir
            )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"处理文件 {other_file} 时发生错误: {str(e)}")
            continue
    
    # 生成汇总报告
    summary = {
        "base_file": base_file,
        "total_comparisons": len(all_results),
        "comparisons": []
    }
    
    for result in all_results:
        summary["comparisons"].append({
            "graph2_name": result["graph2_name"],
            "similarity": result["similarity"],
            "nodes_count_diff": abs(result["graph1_info"]["nodes_count"] - result["graph2_info"]["nodes_count"]),
            "edges_count_diff": abs(result["graph1_info"]["edges_count"] - result["graph2_info"]["edges_count"])
        })
    
    # 按相似度排序
    summary["comparisons"].sort(key=lambda x: x["similarity"], reverse=True)
    
    # 保存汇总报告
    summary_file = os.path.join(output_dir, f"{os.path.splitext(base_file)[0]}_wl_comparison_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总报告已保存到: {summary_file}")
    print("\n相似度排序结果:")
    for i, comp in enumerate(summary["comparisons"], 1):
        print(f"{i:2d}. {comp['graph2_name']}: {comp['similarity']:.4f}")
    
    print(f"\n程序执行完成！所有结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
