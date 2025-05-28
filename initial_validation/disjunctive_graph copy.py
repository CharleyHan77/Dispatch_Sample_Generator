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

class DisjunctiveGraph:
    def __init__(self, parameters: Dict[str, Any]):
        """
        初始化析取图
        :param parameters: 从parser.parse()得到的解析结果
        """
        self.parameters = parameters
        self.jobs = parameters['jobs']
        self.machines = parameters['machinesNb']
        self.graph = nx.DiGraph()
        
        # 添加虚拟开始和结束节点
        self.start_node = 'Start'
        self.end_node = 'End'
        self.graph.add_node(self.start_node, type='virtual')
        self.graph.add_node(self.end_node, type='virtual')
        
        # 生成节点和边
        self._generate_nodes()
        self._generate_edges()
    
    def _generate_nodes(self):
        """生成所有节点"""
        # 添加所有操作节点
        for job_idx, job in enumerate(self.jobs):
            for op_idx in range(len(job)):  # 使用range来确保生成所有工序节点
                node_id = f'J{job_idx+1}O{op_idx+1}'
                # 获取该操作可用的所有机器
                available_machines = [option['machine'] for option in job[op_idx]]  # 机器编从1开始的
                self.graph.add_node(node_id, 
                                  type='operation',
                                  job=job_idx,
                                  operation=op_idx,
                                  available_machines=available_machines)
                
    
    def _generate_edges(self):
        """生成所有边"""
        # 添加作业内的顺序边（conjunctive arcs）
        for job_idx, job in enumerate(self.jobs):
            # 连接起始节点到作业的第一个工序
            first_node = f'J{job_idx+1}O1'
            self.graph.add_edge(self.start_node, first_node, type='conjunctive')
            
            # 连接作业内的相邻工序
            for op_idx in range(len(job)-1):
                current_node = f'J{job_idx+1}O{op_idx+1}'
                next_node = f'J{job_idx+1}O{op_idx+2}'
                self.graph.add_edge(current_node, next_node, type='conjunctive')
            
            # 连接最后一个工序到终止节点
            last_node = f'J{job_idx+1}O{len(job)}'
            self.graph.add_edge(last_node, self.end_node, type='conjunctive')
        
        # 添加析取边（disjunctive arcs）
        # 为每个机器收集可以在其上加工的工序节点
        machine_nodes = {i: [] for i in range(1, self.machines + 1)}  # 机器编号从1开始
        for job_idx, job in enumerate(self.jobs):
            for op_idx, op in enumerate(job):
                node_id = f'J{job_idx+1}O{op_idx+1}'
                # 收集该工序可以使用的所有机器
                for option in op:
                    machine = option['machine']  # 机器编号已经是从1开始的
                    machine_nodes[machine].append(node_id)
        
        # 为每个机器上的工序对添加析取边
        for machine, nodes in machine_nodes.items():
            if len(nodes) > 1:  # 两个或更多节点时，添加析取边
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        # 添加双向析取边
                        self.graph.add_edge(nodes[i], nodes[j], type='disjunctive')
                        self.graph.add_edge(nodes[j], nodes[i], type='disjunctive')
    
    def draw(self, save_path: str = None, title: str = None):
        """
        绘制析取图
        :param save_path: 保存路径
        :param title: 图表标题
        """
        plt.figure(figsize=(30, 24))  # 更大的图形尺寸
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 计算节点位置
        pos = {}
        job_count = len(self.jobs)
        
        # 获取每个作业的最大工序数
        max_ops = max(len(job) for job in self.jobs)
        
        # 首先计算所有作业节点的位置
        for job_idx, job in enumerate(self.jobs):
            y_pos = (job_count - job_idx) * 3.0  # 更大的垂直间距
            for op_idx in range(len(job)):
                node_id = f'J{job_idx+1}O{op_idx+1}'
                # 将工序节点在0.2到0.8之间均匀分布
                x_step = 0.6 / (max_ops + 1)  # 使用0.6作为可用空间（0.8-0.2）
                x_pos = 0.2 + (op_idx + 1) * x_step  # 从0.2开始，确保居中
                pos[node_id] = (x_pos, y_pos)
        
        # 计算所有作业节点的y坐标的最大值和最小值
        y_coords = [coord[1] for coord in pos.values()]
        max_y = max(y_coords)
        min_y = min(y_coords)
        center_y = (max_y + min_y) / 2  # 计算垂直中心位置
        
        # 设置虚拟起始和终止节点的位置（左右两侧）
        pos[self.start_node] = (0.0, center_y)  # 最左侧
        pos[self.end_node] = (1.0, center_y)    # 最右侧
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos,
                             node_color=['red' if node in [self.start_node, self.end_node] else 'lightblue' 
                                       for node in self.graph.nodes()],
                             node_size=2000,  # 增大节点尺寸
                             alpha=0.7)
        
        # 绘制实线（工艺约束）- 有向
        conjunctive_edges = [(u, v) for (u, v, d) in self.graph.edges(data=True) 
                            if d['type'] == 'conjunctive']
        nx.draw_networkx_edges(self.graph, pos,
                             edgelist=conjunctive_edges,
                             edge_color='black',
                             arrows=True,
                             arrowsize=25,
                             width=1.5)  # 增加线条宽度
        
        # 动态生成颜色映射 - 使用更深的颜色
        colors = [
            '#1f77b4',  # 深蓝
            '#d62728',  # 深红
            '#2ca02c',  # 深绿
            '#9467bd',  # 深紫
            '#8c564b',  # 深棕
            '#e377c2',  # 深粉
            '#17becf',  # 青色
            '#ff7f0e',  # 橙色
            '#bcbd22',  # 橄榄绿
            '#7f7f7f',  # 深灰
            '#8B0000',  # 深红色
            '#006400',  # 深绿色
            '#000080',  # 海军蓝
            '#800080',  # 紫色
            '#B8860B',  # 暗金色
            '#B22222',  # 砖红色
            '#4B0082',  # 靛青色
            '#8B4513',  # 马鞍棕色
            '#483D8B',  # 暗板岩蓝
            '#2F4F4F'   # 暗岩灰
        ]

        # 如果机器数量超过预定义颜色，则循环使用
        machine_colors = {i: colors[(i-1) % len(colors)] for i in range(1, self.machines + 1)}
        
        # 为每个机器绘制析取弧（无向虚线）和单节点标记
        legend_elements = []
        for machine, nodes in machine_nodes.items():
            if len(nodes) > 1:  # 两个或更多节点时，绘制析取弧
                # 创建边：将每个节点与下一个节点连接，形成一个封闭的圈
                edges = []
                for i in range(len(nodes)):
                    # 将当前节点与下一个节点连接（循环连接）
                    next_idx = (i + 1) % len(nodes)
                    edges.append((nodes[i], nodes[next_idx]))
                
                if edges:  # 如果有边才绘制
                    # 为不同机器设置不同的弧度，使其更容易区分
                    rad = 0.2 + (machine % 3) * 0.1  # 在0.2到0.4之间变化
                    nx.draw_networkx_edges(self.graph, pos,
                                         edgelist=edges,
                                         edge_color=machine_colors[machine],
                                         style='dashed',
                                         arrows=True,
                                         width=1.2,
                                         alpha=0.6,  # 降低透明度，使重叠的线条更容易看到
                                         connectionstyle=f'arc3,rad={rad}',  # 不同机器使用不同的弧度
                                         arrowsize=10)
            
            elif len(nodes) == 1:  # 只有一个节点时，绘制特殊标记
                node = nodes[0]
                x, y = pos[node]
                # 在节点周围绘制一个带颜色的圆圈标记
                plt.plot(x, y, 'o', 
                        color=machine_colors[machine],
                        markersize=20,
                        fillstyle='none',
                        alpha=0.6)
            
            # 添加到图例
            if len(nodes) > 0:  # 只要有节点就添加到图例
                legend_elements.append(plt.Line2D([0], [0], 
                                               color=machine_colors[machine], 
                                               linestyle='--' if len(nodes) > 1 else 'none',
                                               marker='o' if len(nodes) == 1 else None,
                                               markersize=10 if len(nodes) == 1 else None,
                                               linewidth=1.2,
                                               label=f'M{machine}'))
        
        # 添加节点标签
        labels = {}
        for node in self.graph.nodes():
            if node in [self.start_node, self.end_node]:
                labels[node] = node
            else:
                data = self.graph.nodes[node]
                job_idx = data["job"]
                op_idx = data["operation"]
                # 获取该操作的可选机器列表
                op_machines = []
                for option in self.jobs[job_idx][op_idx]:
                    op_machines.append(f'M{option["machine"]}')
                # 格式化标签，包含作业号、工序号和可选机器列表
                labels[node] = f'J{job_idx+1}O{op_idx+1}\n{", ".join(op_machines)}'
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=14)
        
        # 添加图例
        plt.legend(handles=legend_elements, 
                  title='机器编号',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=12)
        
        if title:
            plt.title(title, fontsize=20, pad=20)
        plt.axis('off')
        
        # 调整布局以适应图例
        plt.subplots_adjust(right=0.85)  # 为图例留出空间
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def generate_colors(n):
    return [f"#{int(h%360):02x}{int((h*50)%255):02x}{int((h*100)%255):02x}" for h in np.linspace(0, 360, n)]

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
            G.add_node(node_name, job=job_idx, op=op_idx, machine=op[0], duration=op[1])
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
    
    # 添加析取约束（虚线）- 同一机器上的工序
    machine_ops = {}  # 记录每个机器上的工序
    for job_idx, job in enumerate(jobs):
        for op_idx, op in enumerate(job):
            machine = op[0]
            if machine not in machine_ops:
                machine_ops[machine] = []
            machine_ops[machine].append((job_idx, op_idx))
    
    # 为同一机器上的工序对添加析取弧
    for machine, ops in machine_ops.items():
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                job1, op1 = ops[i]
                job2, op2 = ops[j]
                op1_name = f'J{job1+1}O{op1+1}'
                op2_name = f'J{job2+1}O{op2+1}'
                G.add_edge(op1_name, op2_name, type='disjunctive')
                G.add_edge(op2_name, op1_name, type='disjunctive')
    
    # 绘制图
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制节点
    nx.draw_networkx_nodes(G, node_positions, 
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.7)
    
    # 分别绘制实线和虚线
    conjunctive_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                        if d.get('type') == 'conjunctive']
    disjunctive_edges = [(u, v) for (u, v, d) in G.edges(data=True) 
                         if d.get('type') == 'disjunctive']
    
    # 绘制实线（工艺约束）
    nx.draw_networkx_edges(G, node_positions, 
                          edgelist=conjunctive_edges,
                          edge_color='black',
                          arrows=True,
                          arrowsize=20)
    
    # 绘制虚线（析取约束）
    nx.draw_networkx_edges(G, node_positions, 
                          edgelist=disjunctive_edges,
                          edge_color='red',
                          style='dashed',
                          arrows=True,
                          arrowsize=20)
    
    # 添加节点标签
    labels = {}
    for node in G.nodes():
        if node in ['Start', 'End']:
            labels[node] = node
        else:
            job = G.nodes[node]['job']
            op = G.nodes[node]['op']
            machine = G.nodes[node]['machine']
            duration = G.nodes[node]['duration']
            labels[node] = f'J{job+1}O{op+1}\nM{machine+1}\n{duration}'
    
    nx.draw_networkx_labels(G, node_positions, labels, font_size=8)
    
    plt.title(f"{title}\n实例: {instance_name}", fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    #return G

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
                    
                    # 创建析取图
                    #dg = DisjunctiveGraph(parameters)
                    
                    
                    # 设置图表标题
                    title = f"{dataset_name} {instance_name} 析取图"
                    
                    # 保存析取图
                    # output_file = os.path.join(output_dir, f"{os.path.splitext(instance_name)[0]}_disjunctive_graph.png")
                    #dg.draw(save_path=output_file, title=title)
                    save_path = os.path.join(output_dir, f"{os.path.splitext(instance_name)[0]}_disjunctive_graph.png")
                    create_disjunctive_graph(parameters, instance_name, save_path, title)
                    print(f"析取图已保存到: save_path")
                    
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