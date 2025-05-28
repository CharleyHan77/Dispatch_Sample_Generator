import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from initial_validation.test_disjunctive_graph_correct import create_disjunctive_graph

def calculate_graph_edit_distance(graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
    """
    计算两个析取图之间的编辑距离
    :param graph1: 第一个析取图
    :param graph2: 第二个析取图
    :return: 编辑距离
    """
    print(f"图1: {len(graph1.nodes())}个节点, {len(graph1.edges())}条边")
    print(f"图2: {len(graph2.nodes())}个节点, {len(graph2.edges())}条边")
    
    # 定义节点匹配成本函数
    def node_match(node1, node2):
        # 如果都是虚拟节点（Start或End），直接返回True
        if node1.get('type') == 'virtual' and node2.get('type') == 'virtual':
            return True
        # 如果节点属性不同，返回False
        if node1.get('job') != node2.get('job') or node1.get('operation') != node2.get('operation'):
            return False
        # 比较可用机器和加工时间
        if set(node1.get('available_machines', [])) != set(node2.get('available_machines', [])):
            return False
        if node1.get('processing_times', []) != node2.get('processing_times', []):
            return False
        return True

    # 定义边匹配成本函数
    def edge_match(edge1, edge2):
        return edge1.get('type') == edge2.get('type')

    print("开始计算编辑距离...")
    start_time = time.time()
    try:
        # 使用networkx的graph_edit_distance函数，设置超时时间
        distance = nx.graph_edit_distance(graph1, graph2, 
                                        node_match=node_match,
                                        edge_match=edge_match,
                                        timeout=30)  # 设置30秒超时
        end_time = time.time()
        print(f"编辑距离计算完成，耗时: {end_time - start_time:.2f}秒")
        return distance
    except nx.NetworkXTimeoutError:
        end_time = time.time()
        print(f"计算超时，返回近似值，耗时: {end_time - start_time:.2f}秒")
        # 如果超时，返回一个近似值
        return float('inf')
    except Exception as e:
        end_time = time.time()
        print(f"计算过程中出现错误: {str(e)}，耗时: {end_time - start_time:.2f}秒")
        return float('inf')

def compare_two_instances(instance1_path: str, instance2_path: str, save_path: Optional[str] = None) -> float:
    """
    比较两个实例的析取图
    :param instance1_path: 第一个实例的路径
    :param instance2_path: 第二个实例的路径
    :param save_path: 保存比较结果的路径（可选）
    :return: 编辑距离
    """
    print(f"\n比较实例: {os.path.basename(instance1_path)} 和 {os.path.basename(instance2_path)}")
    
    # 解析两个实例
    parameters1 = parser.parse(instance1_path)
    print(f"parameters1内容: {parameters1}")
    parameters2 = parser.parse(instance2_path)
    
    # 创建两个析取图
    G1 = create_disjunctive_graph(parameters1, os.path.basename(instance1_path), None, "图1")
    if G1 is None:
        raise ValueError(f"无法创建实例 {os.path.basename(instance1_path)} 的析取图")
    
    G2 = create_disjunctive_graph(parameters2, os.path.basename(instance2_path), None, "图2")
    if G2 is None:
        raise ValueError(f"无法创建实例 {os.path.basename(instance2_path)} 的析取图")
    
    # 计算编辑距离
    start_time = time.time()
    distance = calculate_graph_edit_distance(G1, G2)
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"实例对比计算总耗时: {computation_time:.2f}秒")
    
    # 如果提供了保存路径，保存比较结果
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"实例1: {os.path.basename(instance1_path)}\n")
            f.write(f"实例2: {os.path.basename(instance2_path)}\n")
            f.write(f"编辑距离: {distance}\n")
            f.write(f"计算耗时: {computation_time:.2f}秒\n")
            f.write(f"\n图1信息:\n")
            f.write(f"节点数: {len(G1.nodes())}\n")
            f.write(f"工艺约束边数: {len([e for e in G1.edges(data=True) if e[2].get('type') == 'conjunctive'])}\n")
            f.write(f"析取弧数: {len([e for e in G1.edges(data=True) if e[2].get('type') == 'disjunctive'])}\n")
            f.write(f"总边数: {len(G1.edges())}\n")
            f.write(f"\n图2信息:\n")
            f.write(f"节点数: {len(G2.nodes())}\n")
            f.write(f"工艺约束边数: {len([e for e in G2.edges(data=True) if e[2].get('type') == 'conjunctive'])}\n")
            f.write(f"析取弧数: {len([e for e in G2.edges(data=True) if e[2].get('type') == 'disjunctive'])}\n")
            f.write(f"总边数: {len(G2.edges())}\n")
    
    return distance

def main():
    """主函数"""
    try:
        print("开始运行析取图比较程序...")
        total_start_time = time.time()
        
        # 获取数据集根目录
        dataset_root = os.path.join(project_root, "dataset", "Brandimarte")
        print(f"数据集目录: {dataset_root}")
        
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")
        
        # 创建输出目录
        output_dir = os.path.join(project_root, "output", "graph_comparisons")
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有.fjs文件
        instance_files = [f for f in os.listdir(dataset_root) if f.endswith('.fjs')]
        
        # 比较所有可能的实例对
        results = []
        for i in range(len(instance_files)):
            for j in range(i + 1, len(instance_files)):
                instance1 = instance_files[i]
                instance2 = instance_files[j]
                
                instance1_path = os.path.join(dataset_root, instance1)
                instance2_path = os.path.join(dataset_root, instance2)
                
                # 设置保存路径
                save_path = os.path.join(output_dir, f"{os.path.splitext(instance1)[0]}_{os.path.splitext(instance2)[0]}_comparison.txt")
                
                # 计算编辑距离
                distance = compare_two_instances(instance1_path, instance2_path, save_path)
                results.append((instance1, instance2, distance))
        
        # 按编辑距离排序结果
        results.sort(key=lambda x: x[2])
        
        # 保存总体比较结果
        summary_path = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("实例对比较结果（按编辑距离排序）：\n\n")
            for instance1, instance2, distance in results:
                f.write(f"{instance1} - {instance2}: {distance}\n")
        
        total_end_time = time.time()
        print(f"\n比较结果已保存到: {summary_path}")
        print(f"程序总运行时间: {total_end_time - total_start_time:.2f}秒")
        
    except Exception as e:
        print("程序执行过程中发生严重错误:")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 