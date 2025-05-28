import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib.font_manager as fm
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from feature_extraction.feature_extractor import FeatureExtractor
from initial_validation.utils import parser

def extract_features_from_file(file_path):
    """
    从fjs文件中提取特征
    """
    parameters = parser.parse(file_path)
    extractor = FeatureExtractor(parameters)
    features = extractor.extract_all_features()
    return features

def create_feature_correlation_graph(features, instance_name, save_path=None, title="特征相关性图"):
    """
    创建单个实例的特征相关性图
    :param features: 特征字典
    :param instance_name: 实例名称
    :param save_path: 保存路径
    :param title: 图表标题
    """
    # 将特征字典转换为扁平化的DataFrame
    flat_features = {}
    for category, category_features in features.items():
        for feature_name, feature_value in category_features.items():
            flat_features[f"{category}_{feature_name}"] = feature_value
    
    df = pd.DataFrame([flat_features])
    
    # 创建图
    G = nx.Graph()
    
    # 添加节点
    feature_names = list(df.columns)
    for feature_name in feature_names:
        G.add_node(feature_name, value=df[feature_name].iloc[0])
    
    # 计算特征之间的相似度（使用特征值的差异）
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            # 计算特征值的相对差异
            val1 = df[feature_names[i]].iloc[0]
            val2 = df[feature_names[j]].iloc[0]
            if val1 != 0 and val2 != 0:  # 避免除以零
                similarity = 1 - abs(val1 - val2) / max(abs(val1), abs(val2))
                if similarity > 0.3:  # 只添加相似度大于0.3的边
                    G.add_edge(feature_names[i], feature_names[j], 
                              weight=similarity)
    
    # 绘制图
    plt.figure(figsize=(15, 12))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用spring_layout布局，增加k值使节点分布更分散
    pos = nx.spring_layout(G, k=2, seed=42)
    
    # 绘制节点，使用特征值的大小来调整节点大小
    node_sizes = [abs(G.nodes[node]['value']) * 500 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7)
    
    # 绘制边，根据权重调整边的宽度
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                          edge_color='gray')
    
    # 添加标签，调整字体大小
    labels = {}
    for node in G.nodes():
        node_name = node.replace('_', '\n')
        value = G.nodes[node]['value']
        labels[node] = f"{node_name}\n{value:.2f}"
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # 添加边权重标签
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                  for u, v in G.edges() 
                  if G[u][v]['weight'] > 0.5}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title(f"{title}\n实例: {instance_name}", fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return G

def process_instance(file_path, output_dir):
    """
    处理单个实例文件
    :param file_path: 实例文件路径
    :param output_dir: 输出目录
    """
    try:
        # 获取数据集名称和实例名称
        dataset_name = os.path.basename(os.path.dirname(file_path))
        instance_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 创建数据集输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 提取特征
        features = extract_features_from_file(file_path)
        
        # 生成时间戳
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建特征图
        save_path = os.path.join(dataset_output_dir, f"{instance_name}_feature_graph.png")
        print(f"正在生成特征图: {save_path}")
        
        create_feature_correlation_graph(
            features,
            instance_name,
            save_path,
            f"{instance_name}特征相关性图"
        )
        
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def process_dataset(dataset_dir, output_dir):
    """
    处理数据集目录下的所有.fjs文件
    :param dataset_dir: 数据集目录
    :param output_dir: 输出目录
    """
    success_count = 0
    fail_count = 0
    
    # 遍历数据集目录
    for root, dirs, files in os.walk(dataset_dir):
        fjs_files = [f for f in files if f.endswith('.fjs')]
        if not fjs_files:
            continue
        
        dataset_name = os.path.basename(root)
        if dataset_name == "Text":
            dataset_name = os.path.basename(os.path.dirname(root))
        
        print(f"\n正在处理数据集: {dataset_name}")
        
        # 处理每个实例
        for fjs_file in fjs_files:
            file_path = os.path.join(root, fjs_file)
            if process_instance(file_path, output_dir):
                success_count += 1
            else:
                fail_count += 1
    
    print(f"\n处理完成！成功: {success_count}个实例，失败: {fail_count}个实例")

def main():
    # 创建输出目录
    output_dir = os.path.join(project_root, "output", "feature_graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理数据集目录
    dataset_dir = os.path.join(project_root, "dataset")
    process_dataset(dataset_dir, output_dir)

if __name__ == "__main__":
    main()