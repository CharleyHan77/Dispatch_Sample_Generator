import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser

def create_disjunctive_graph(parameters: Dict[str, Any], instance_name: str, save_path: Optional[str] = None, title: str = "析取图") -> None:
    """
    创建FJSP析取图的基础结构（节点和实线部分）
    :param parameters: 从parser.parse()得到的解析结果
    :param instance_name: 实例名称
    :param save_path: 保存路径
    :param title: 图表标题
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 获取作业数和机器数
    jobs = parameters['jobs']
    job_count = len(jobs)
    machines_count = parameters['machinesNb']
    
    # 计算节点位置
    node_positions = {}
    
    # 为每个作业的工序分配位置
    for job_idx, job in enumerate(jobs):
        # 将作业的所有工序排在一行
        y_pos = job_count - job_idx  # 从上到下排列作业
        for op_idx, op in enumerate(job):
            # 在0到1之间均匀分布工序
            x_pos = (op_idx + 1) / (len(job) + 1)  # 添加+1使节点更居中
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
            node_positions[node_name] = (x_pos, y_pos)
    
    # 添加虚拟起始节点和终止节点
    node_positions['Start'] = (0.0, (job_count + 1) / 2)  # 左侧居中
    node_positions['End'] = (1.0, (job_count + 1) / 2)    # 右侧居中
    G.add_node('Start', type='virtual')
    G.add_node('End', type='virtual')
    
    # 添加工艺约束（实线）
    for job_idx, job in enumerate(jobs):
        # 连接起始节点到作业的第一个工序
        first_op = f'J{job_idx+1}O1'
        G.add_edge('Start', first_op, type='conjunctive')
        
        # 连接作业内的相邻工序
        for op_idx in range(len(job)-1):
            current_op = f'J{job_idx+1}O{op_idx+1}'
            next_op = f'J{job_idx+1}O{op_idx+2}'
            G.add_edge(current_op, next_op, type='conjunctive')
        
        # 连接最后一个工序到终止节点
        last_op = f'J{job_idx+1}O{len(job)}'
        G.add_edge(last_op, 'End', type='conjunctive')
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制节点
    nx.draw_networkx_nodes(G, node_positions, 
                          node_color=['red' if node in ['Start', 'End'] else 'lightblue' 
                                    for node in G.nodes()],
                          node_size=500,
                          alpha=0.7)
    
    # 绘制实线（工艺约束）
    conjunctive_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d.get('type') == 'conjunctive']
    nx.draw_networkx_edges(G, node_positions, 
                          edgelist=conjunctive_edges,
                          edge_color='black',
                          arrows=True,
                          arrowsize=8,
                          width=1.5)
    
    # 添加节点标签
    labels = {}
    for node in G.nodes():
        if node in ['Start', 'End']:
            labels[node] = node
        else:
            data = G.nodes[node]
            job = data['job']
            op = data['operation']
            machines = [f'M{m}' for m in data['available_machines']]
            times = data['processing_times']
            # 格式化标签：作业号、工序号、可用机器和对应加工时间
            machine_info = [f'{m}({t})' for m, t in zip(machines, times)]
            labels[node] = f'J{job}O{op}\n{", ".join(machine_info)}'
    
    nx.draw_networkx_labels(G, node_positions, labels, font_size=6)
    
    plt.title(f"{title}\n实例: {instance_name}", fontsize=16, pad=20)
    plt.axis('off')
    
    # 调整布局以确保图形完整显示
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """主函数"""
    try:
        print("开始运行析取图生成程序...")
        
        # 获取数据集根目录
        dataset_root = os.path.join(project_root, "test_dataset")
        print(f"数据集根目录: {dataset_root}")
        
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")
        
        # 统计信息
        total_files = 0
        successful_files = 0
        failed_files = 0
        
        # 遍历数据集目录
        print("\n开始处理数据集...")
        for dataset_name in os.listdir(dataset_root):
            dataset_path = os.path.join(dataset_root, dataset_name)
            print(f"\n处理数据集: {dataset_name}")
            
            # 确保是目录
            if not os.path.isdir(dataset_path):
                print(f"跳过非目录项: {dataset_name}")
                continue
                
            # 创建对应的输出目录
            output_dir = os.path.join(project_root, "test_output", "test_disjunctive_graphs", dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
            
            # 处理该数据集下的所有.fjs文件
            for instance_name in os.listdir(dataset_path):
                if not instance_name.endswith('.fjs'):
                    continue
                    
                total_files += 1
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
                    successful_files += 1
                    
                except Exception as e:
                    print(f"处理文件 {instance_name} 时发生错误:")
                    print(str(e))
                    failed_files += 1
        
        # 输出统计信息
        print("\n处理完成统计:")
        print(f"总文件数: {total_files}")
        print(f"成功处理: {successful_files}")
        print(f"处理失败: {failed_files}")
        
    except Exception as e:
        print("程序执行过程中发生严重错误:")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 