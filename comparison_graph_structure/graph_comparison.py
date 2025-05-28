import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import networkx as nx
from collections import defaultdict
from networkx.algorithms.isomorphism import DiGraphMatcher

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from initial_validation.disjunctive_graph import DisjunctiveGraph

class GraphComparator:
    def __init__(self):
        """初始化图比较器"""
        pass
    
    def preprocess_graph(self, dg: DisjunctiveGraph) -> nx.MultiDiGraph:
        """将析取图转换为规范化形式，保留拓扑和约束类型"""
        G = nx.MultiDiGraph()  # 使用多重有向图区分边类型
        
        # 添加所有节点（保留类型属性）
        for node, data in dg.graph.nodes(data=True):
            # 如果节点没有type属性，根据节点名称判断类型
            if 'type' not in data:
                if node in ['Start', 'End']:
                    node_type = 'virtual'
                else:
                    node_type = 'operation'
            else:
                node_type = data['type']
            G.add_node(node, type=node_type)
        
        # 添加边并标记类型
        for u, v, data in dg.graph.edges(data=True):
            # 如果边没有type属性，根据节点类型判断边类型
            if 'type' not in data:
                if u == 'Start' or v == 'End' or (u.startswith('J') and v.startswith('J') and u[1] == v[1]):
                    edge_type = 'conjunctive'
                else:
                    edge_type = 'disjunctive'
            else:
                edge_type = data['type']
            G.add_edge(u, v, type=edge_type)
        
        return G
    
    def check_isomorphism(self, dg1: nx.MultiDiGraph, dg2: nx.MultiDiGraph) -> bool:
        """检查两个析取图是否拓扑同构"""
        # 节点匹配函数（忽略具体工件/工序编号，只比较类型）
        def node_match(n1, n2):
            return n1['type'] == n2['type']
        
        # 边匹配函数（必须类型相同）
        def edge_match(e1, e2):
            return e1['type'] == e2['type']
        
        return DiGraphMatcher(dg1, dg2, node_match, edge_match).is_isomorphic()
    
    def find_mcs(self, dg1: nx.MultiDiGraph, dg2: nx.MultiDiGraph) -> float:
        """查找最大公共子图并返回相似度得分"""
        def node_match(n1, n2):
            return n1['type'] == n2['type']
        
        def edge_match(e1, e2):
            return e1['type'] == e2['type']
        
        matcher = DiGraphMatcher(dg1, dg2, node_match, edge_match)
        mcs = max(matcher.subgraph_isomorphisms_iter(), key=len, default={})
        
        # 计算相似度（基于节点重叠率）
        union_size = max(len(dg1), len(dg2))
        return len(mcs) / union_size if union_size > 0 else 0
    
    def compare_constraints(self, dg1: nx.MultiDiGraph, dg2: nx.MultiDiGraph) -> float:
        """比较析取弧的分布模式"""
        def extract_disjunctive_pattern(G):
            pattern = defaultdict(int)
            for u, v, data in G.edges(data=True):
                if data['type'] == 'disjunctive':
                    # 统计机器冲突模式
                    pattern[(u, v)] += 1
            return pattern
        
        p1 = extract_disjunctive_pattern(dg1)
        p2 = extract_disjunctive_pattern(dg2)
        
        # 计算Jaccard相似度
        common = sum(min(p1[k], p2.get(k, 0)) for k in p1)
        total = sum(p1.values()) + sum(p2.values()) - common
        return common / total if total > 0 else 0
    
    def graph_similarity(self, dg1: nx.MultiDiGraph, dg2: nx.MultiDiGraph, 
                        weights: List[float] = [0.4, 0.3, 0.3]) -> float:
        """综合拓扑相似性评分"""
        # 1. 同构检测（二元指标）
        iso_score = 1.0 if self.check_isomorphism(dg1, dg2) else 0
        
        # 2. MCS大小
        mcs_score = self.find_mcs(dg1, dg2)
        
        # 3. 约束冲突相似度
        constraint_score = self.compare_constraints(dg1, dg2)
        
        return (weights[0] * iso_score + 
                weights[1] * mcs_score + 
                weights[2] * constraint_score)

def main():
    try:
        # 创建输出目录
        output_dir = os.path.join(project_root, "output", "comparison_graph_result")
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化比较器
        comparator = GraphComparator()
        
        # 加载新生成的析取图
        new_data_file = os.path.join(project_root, "new_data", "new_fjsp_data.fjs")
        new_parameters = parser.parse(new_data_file)
        new_dg = DisjunctiveGraph(new_parameters)
        new_graph = comparator.preprocess_graph(new_dg)
        
        # 结果字典
        results = {}
        
        # 遍历所有数据集目录
        dataset_root = os.path.join(project_root, "dataset")
        for dataset_name in os.listdir(dataset_root):
            dataset_path = os.path.join(dataset_root, dataset_name)
            
            # 确保是目录
            if not os.path.isdir(dataset_path):
                continue
            
            # 处理该数据集下的所有.fjs文件
            for instance_name in os.listdir(dataset_path):
                if not instance_name.endswith('.fjs'):
                    continue
                
                file_path = os.path.join(dataset_path, instance_name)
                print(f"\n正在比较: {instance_name}")
                
                try:
                    # 解析数据文件
                    parameters = parser.parse(file_path)
                    
                    # 创建析取图
                    dg = DisjunctiveGraph(parameters)
                    graph = comparator.preprocess_graph(dg)
                    
                    # 计算相似度
                    similarity = comparator.graph_similarity(new_graph, graph)
                    
                    # 保存结果
                    results[f"{dataset_name}/{instance_name}"] = {
                        "similarity": similarity,
                        "is_isomorphic": comparator.check_isomorphism(new_graph, graph),
                        "mcs_score": comparator.find_mcs(new_graph, graph),
                        "constraint_score": comparator.compare_constraints(new_graph, graph)
                    }
                    
                    print(f"相似度: {similarity:.3f}")
                    
                except Exception as e:
                    print(f"处理文件 {instance_name} 时发生错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 保存结果到JSON文件
        output_file = os.path.join(output_dir, "comparison_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n比较结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 