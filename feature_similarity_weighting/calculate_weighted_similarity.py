import os
import json
import numpy as np
from scipy.stats import entropy
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
import sys
sys.path.append(project_root)

# 导入析取图相关函数
from comparison_disjunctive_graphs.compare_graphs_wl import (
    create_disjunctive_graph_with_attributes,
    init_node_labels,
    add_edge_attributes,
    wl_step
)
from initial_validation.utils import parser

# 配置日志
def setup_logger():
    """设置日志记录器"""
    # 创建logs目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"weighted_similarity_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger("weighted_similarity")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def normalize_features(features_dict):
    """对特征进行标准化处理"""
    # 提取所有特征值
    basic_values = []
    processing_values = []
    
    for file_features in features_dict.values():
        # 基础特征
        basic_features = file_features['basic_features']
        basic_values.append([
            basic_features['num_jobs'],
            basic_features['num_machines'],
            basic_features['total_operations'],
            basic_features['avg_available_machines'],
            basic_features['std_available_machines']
        ])
        
        # 加工时间特征
        processing_features = file_features['processing_time_features']
        processing_values.append([
            processing_features['processing_time_mean'],
            processing_features['processing_time_std'],
            processing_features['processing_time_min'],
            processing_features['processing_time_max'],
            processing_features['machine_time_variance']
        ])
    
    # 转换为numpy数组
    basic_values = np.array(basic_values)
    processing_values = np.array(processing_values)
    
    # 计算每个特征的均值和标准差
    basic_means = np.mean(basic_values, axis=0)
    basic_stds = np.std(basic_values, axis=0)
    processing_means = np.mean(processing_values, axis=0)
    processing_stds = np.std(processing_values, axis=0)
    
    # 添加小的常数避免0标准差
    epsilon = 1e-10
    basic_stds = np.where(basic_stds == 0, epsilon, basic_stds)
    processing_stds = np.where(processing_stds == 0, epsilon, processing_stds)
    
    # 标准化所有特征
    normalized_features = {}
    for file_name, file_features in features_dict.items():
        normalized_features[file_name] = {
            'basic_features': {},
            'processing_time_features': {}
        }
        
        # 标准化基础特征
        basic_features = file_features['basic_features']
        normalized_features[file_name]['basic_features'] = {
            'num_jobs': (basic_features['num_jobs'] - basic_means[0]) / basic_stds[0],
            'num_machines': (basic_features['num_machines'] - basic_means[1]) / basic_stds[1],
            'total_operations': (basic_features['total_operations'] - basic_means[2]) / basic_stds[2],
            'avg_available_machines': (basic_features['avg_available_machines'] - basic_means[3]) / basic_stds[3],
            'std_available_machines': (basic_features['std_available_machines'] - basic_means[4]) / basic_stds[4]
        }
        
        # 标准化加工时间特征
        processing_features = file_features['processing_time_features']
        normalized_features[file_name]['processing_time_features'] = {
            'processing_time_mean': (processing_features['processing_time_mean'] - processing_means[0]) / processing_stds[0],
            'processing_time_std': (processing_features['processing_time_std'] - processing_means[1]) / processing_stds[1],
            'processing_time_min': (processing_features['processing_time_min'] - processing_means[2]) / processing_stds[2],
            'processing_time_max': (processing_features['processing_time_max'] - processing_means[3]) / processing_stds[3],
            'machine_time_variance': (processing_features['machine_time_variance'] - processing_means[4]) / processing_stds[4]
        }
    
    return normalized_features

def calculate_euclidean_distance(features1, features2):
    """计算两个特征向量之间的欧氏距离"""
    vec1 = np.array(list(features1.values()))
    vec2 = np.array(list(features2.values()))
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def calculate_js_divergence(p, q):
    """计算两个概率分布之间的JS散度"""
    # 确保概率分布和为1
    p = np.array(p)
    q = np.array(q)
    
    # 添加小的常数避免0值
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 计算平均分布
    m = 0.5 * (p + q)
    
    # 计算JS散度
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    
    # 处理可能的无穷大值
    if np.isinf(js_div):
        return 1.0  # 返回最大距离
    
    return js_div

def normalize_distance(distance, max_distance):
    """将距离归一化到[0,1]区间，并转换为相似度（1-归一化距离）"""
    if max_distance <= 0:
        return 1.0
    return 1 - (distance / max_distance)

def calculate_disjunctive_graph_similarity(graph_info1, graph_info2):
    """计算两个析取图的相似度，基于图结构特征和WL标签频率"""
    # 获取图的基本结构特征
    nodes1 = graph_info1['nodes_count']
    nodes2 = graph_info2['nodes_count']
    edges1 = graph_info1['edges_count']
    edges2 = graph_info2['edges_count']
    
    # 计算结构相似度（基于节点数和边数的相似性）
    nodes_similarity = 1 - abs(nodes1 - nodes2) / max(nodes1, nodes2)
    edges_similarity = 1 - abs(edges1 - edges2) / max(edges1, edges2)
    structure_similarity = (nodes_similarity + edges_similarity) / 2
    
    # 获取实线和虚线标签频率
    solid_freq1 = graph_info1['solid_frequency']
    solid_freq2 = graph_info2['solid_frequency']
    dashed_freq1 = graph_info1['dashed_frequency']
    dashed_freq2 = graph_info2['dashed_frequency']
    
    # 计算实线标签的Jaccard相似度
    solid_keys1 = set(solid_freq1.keys())
    solid_keys2 = set(solid_freq2.keys())
    if len(solid_keys1.union(solid_keys2)) > 0:
        solid_jaccard = len(solid_keys1.intersection(solid_keys2)) / len(solid_keys1.union(solid_keys2))
    else:
        solid_jaccard = 0.0
    
    # 计算虚线标签的Jaccard相似度
    dashed_keys1 = set(dashed_freq1.keys())
    dashed_keys2 = set(dashed_freq2.keys())
    if len(dashed_keys1.union(dashed_keys2)) > 0:
        dashed_jaccard = len(dashed_keys1.intersection(dashed_keys2)) / len(dashed_keys1.union(dashed_keys2))
    else:
        dashed_jaccard = 0.0
    
    # 计算标签分布相似度（使用余弦相似度）
    all_solid_keys = solid_keys1.union(solid_keys2)
    solid_vec1 = np.array([solid_freq1.get(k, 0) for k in all_solid_keys])
    solid_vec2 = np.array([solid_freq2.get(k, 0) for k in all_solid_keys])
    
    solid_norm1 = np.linalg.norm(solid_vec1)
    solid_norm2 = np.linalg.norm(solid_vec2)
    
    if solid_norm1 == 0 or solid_norm2 == 0:
        solid_cosine = 0.0
    else:
        solid_cosine = np.dot(solid_vec1, solid_vec2) / (solid_norm1 * solid_norm2)
    
    all_dashed_keys = dashed_keys1.union(dashed_keys2)
    dashed_vec1 = np.array([dashed_freq1.get(k, 0) for k in all_dashed_keys])
    dashed_vec2 = np.array([dashed_freq2.get(k, 0) for k in all_dashed_keys])
    
    dashed_norm1 = np.linalg.norm(dashed_vec1)
    dashed_norm2 = np.linalg.norm(dashed_vec2)
    
    if dashed_norm1 == 0 or dashed_norm2 == 0:
        dashed_cosine = 0.0
    else:
        dashed_cosine = np.dot(dashed_vec1, dashed_vec2) / (dashed_norm1 * dashed_norm2)
    
    # 综合相似度计算
    # 调整权重分配，增加结构相似度的影响
    solid_similarity = 0.3 * solid_jaccard + 0.7 * solid_cosine
    dashed_similarity = 0.3 * dashed_jaccard + 0.7 * dashed_cosine
    
    # 加权组合相似度（实线权重0.6，虚线权重0.4）
    label_similarity = 0.6 * solid_similarity + 0.4 * dashed_similarity
    
    # 最终相似度：增加结构相似度权重，减少标签相似度权重
    weighted_similarity = 0.5 * structure_similarity + 0.5 * label_similarity
    
    return weighted_similarity

def visualize_top_5_similarity(top_5_results, output_dir, new_data_file):
    """可视化前5名相似文件的详细对比"""
    # 准备数据
    files = [file for file, _ in top_5_results]
    basic_similarities = [results['basic_similarity'] for _, results in top_5_results]
    processing_similarities = [results['processing_similarity'] for _, results in top_5_results]
    kde_similarities = [results['kde_similarity'] for _, results in top_5_results]
    disjunctive_similarities = [results['disjunctive_similarity'] for _, results in top_5_results]
    weighted_similarities = [results['weighted_similarity'] for _, results in top_5_results]
    
    # 创建图表
    plt.figure(figsize=(15, 7))
    
    # 设置柱状图的位置
    x = np.arange(len(files))
    width = 0.15
    
    # 绘制五种相似度
    plt.bar(x - width*2, basic_similarities, width, label='基础特征相似度', color='#2ecc71')
    plt.bar(x - width*1, processing_similarities, width, label='加工时间特征相似度', color='#3498db')
    plt.bar(x, kde_similarities, width, label='KDE相似度', color='#9b59b6')
    plt.bar(x + width*1, disjunctive_similarities, width, label='析取图相似度', color='#f39c12')
    plt.bar(x + width*2, weighted_similarities, width, label='综合加权相似度', color='#e74c3c')
    
    # 为每种相似度添加数值标签
    for i, v in enumerate(basic_similarities):
        plt.text(i - width*2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(processing_similarities):
        plt.text(i - width*1, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(kde_similarities):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(disjunctive_similarities):
        plt.text(i + width*1, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(weighted_similarities):
        plt.text(i + width*2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 设置图表属性
    plt.xlabel('历史数据文件')
    plt.ylabel('相似度')
    plt.title(f'新数据 {new_data_file} 的前5名相似fjs数据对比')
    plt.xticks(x, files, rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围，让数据对比更明显
    min_similarity = min(min(basic_similarities), min(processing_similarities), 
                        min(kde_similarities), min(disjunctive_similarities), min(weighted_similarities))
    max_similarity = max(max(basic_similarities), max(processing_similarities), 
                        max(kde_similarities), max(disjunctive_similarities), max(weighted_similarities))
    
    # 计算合适的y轴范围
    y_min = max(0, min_similarity - 0.01)  # 从最小值减0.01开始
    y_max = min(1, max_similarity + 0.01)  # 到最大值加0.01结束
    
    # 设置y轴刻度的精度（显示2位小数）
    y_ticks = np.arange(y_min, y_max + 0.001, 0.01)  # 使用0.01的间隔
    plt.yticks(y_ticks, [f'{y:.2f}' for y in y_ticks])  # 显示2位小数
    
    # 设置y轴范围
    plt.ylim(y_min, y_max)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, 'top_5_similarity_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logger = setup_logger()
    logger.info("开始计算综合加权相似度...")
    
    # 记录总开始时间
    total_start_time = time.time()
    
    try:
        # 获取当前目录下的所有new_data_*目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dirs = [d for d in os.listdir(current_dir) if d.startswith('new_data_') and os.path.isdir(os.path.join(current_dir, d))]
        
        for data_dir in data_dirs:
            logger.info(f"\n处理目录: {data_dir}")
            
            # 记录数据加载开始时间
            load_start_time = time.time()
            
            # 加载新数据特征
            new_data_features_path = os.path.join("feature_similarity_weighting", data_dir, "new_data_features.json")
            with open(new_data_features_path, 'r', encoding='utf-8') as f:
                new_data_features = json.load(f)
            
            # 加载历史数据特征
            historical_features_path = os.path.join("output", "dataset_features.json")
            with open(historical_features_path, 'r', encoding='utf-8') as f:
                historical_features = json.load(f)
            
            # 记录数据加载时间
            load_time = time.time() - load_start_time
            logger.info(f"数据加载完成，耗时: {load_time:.2f}秒")
            
            # 获取新数据文件名
            new_data_file = list(new_data_features.keys())[0]
            logger.info(f"处理新数据文件: {new_data_file}")
            
            # 记录特征标准化开始时间
            normalize_start_time = time.time()
            
            # 合并所有特征用于标准化
            all_features = {**historical_features, **new_data_features}
            
            # 标准化特征
            logger.info("正在标准化特征...")
            normalized_features = normalize_features(all_features)
            
            # 记录特征标准化时间
            normalize_time = time.time() - normalize_start_time
            logger.info(f"特征标准化完成，耗时: {normalize_time:.2f}秒")
            
            # 存储所有相似度结果
            similarity_results = {}
            
            # 记录最大距离计算开始时间
            max_dist_start_time = time.time()
            
            # 计算最大距离（用于归一化）
            max_basic_distance = 0
            max_processing_distance = 0
            
            # 第一遍：计算最大距离
            for hist_file, hist_features in historical_features.items():
                # 基础特征距离
                basic_distance = calculate_euclidean_distance(
                    normalized_features[new_data_file]["basic_features"],
                    normalized_features[hist_file]["basic_features"]
                )
                max_basic_distance = max(max_basic_distance, basic_distance)
                
                # 加工时间特征距离
                processing_distance = calculate_euclidean_distance(
                    normalized_features[new_data_file]["processing_time_features"],
                    normalized_features[hist_file]["processing_time_features"]
                )
                max_processing_distance = max(max_processing_distance, processing_distance)
            
            # 记录最大距离计算时间
            max_dist_time = time.time() - max_dist_start_time
            logger.info(f"最大距离计算完成，耗时: {max_dist_time:.2f}秒")
            
            # 记录相似度计算开始时间
            similarity_start_time = time.time()
            
            # 第二遍：计算综合相似度
            for hist_file, hist_features in historical_features.items():
                logger.info(f"\n比较历史数据文件: {hist_file}")
                
                # 1. 计算基础特征相似度
                basic_distance = calculate_euclidean_distance(
                    normalized_features[new_data_file]["basic_features"],
                    normalized_features[hist_file]["basic_features"]
                )
                basic_similarity = normalize_distance(basic_distance, max_basic_distance)
                logger.info(f"基础特征相似度: {basic_similarity:.4f}")
                
                # 2. 计算加工时间特征相似度
                processing_distance = calculate_euclidean_distance(
                    normalized_features[new_data_file]["processing_time_features"],
                    normalized_features[hist_file]["processing_time_features"]
                )
                processing_similarity = normalize_distance(processing_distance, max_processing_distance)
                logger.info(f"加工时间特征相似度: {processing_similarity:.4f}")
                
                # 3. 计算KDE相似度（JS散度）
                kde_similarity = 1 - calculate_js_divergence(
                    new_data_features[new_data_file]["kde_features"]["density"],
                    historical_features[hist_file]["kde_features"]["density"]
                )
                logger.info(f"KDE相似度: {kde_similarity:.4f}")
                
                # 4. 计算析取图相似度
                disjunctive_similarity = calculate_disjunctive_graph_similarity(
                    new_data_features[new_data_file]["disjunctive_graphs_features"],
                    historical_features[hist_file]["disjunctive_graphs_features"]
                )
                logger.info(f"析取图相似度: {disjunctive_similarity:.4f}")
                
                # 5. 计算综合加权相似度（新权重分配）
                weighted_similarity = (
                    0.3 * basic_similarity +
                    0.25 * processing_similarity +
                    0.2 * kde_similarity +
                    0.25 * disjunctive_similarity
                )
                logger.info(f"综合加权相似度: {weighted_similarity:.4f}")
                
                # 存储结果
                similarity_results[hist_file] = {
                    "basic_similarity": basic_similarity,
                    "processing_similarity": processing_similarity,
                    "kde_similarity": kde_similarity,
                    "disjunctive_similarity": disjunctive_similarity,
                    "weighted_similarity": weighted_similarity
                }
            
            # 记录相似度计算时间
            similarity_time = time.time() - similarity_start_time
            logger.info(f"相似度计算完成，耗时: {similarity_time:.2f}秒")
            
            # 记录结果保存开始时间
            save_start_time = time.time()
            
            # 按综合加权相似度排序
            sorted_results = sorted(
                similarity_results.items(),
                key=lambda x: x[1]["weighted_similarity"],
                reverse=True
            )
            
            # 获取前5名文件
            top_5_results = sorted_results[:5]
            
            # 保存结果
            output_dir = os.path.join("feature_similarity_weighting", data_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "similarity_results.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(similarity_results, f, indent=4, ensure_ascii=False)
            
            # 生成前5名相似文件的对比图表
            logger.info("正在生成前5名相似文件的对比图表...")
            visualize_top_5_similarity(top_5_results, output_dir, new_data_file)
            
            # 记录结果保存时间
            save_time = time.time() - save_start_time
            logger.info(f"结果保存完成，耗时: {save_time:.2f}秒")
            
            # 输出最相似的前5个结果
            logger.info("\n最相似的前5个历史数据文件：")
            for i, (file_name, scores) in enumerate(top_5_results, 1):
                logger.info(f"\n{i}. {file_name}")
                logger.info(f"   基础特征相似度: {scores['basic_similarity']:.4f}")
                logger.info(f"   加工时间特征相似度: {scores['processing_similarity']:.4f}")
                logger.info(f"   KDE相似度: {scores['kde_similarity']:.4f}")
                logger.info(f"   析取图相似度: {scores['disjunctive_similarity']:.4f}")
                logger.info(f"   综合加权相似度: {scores['weighted_similarity']:.4f}")
            
            logger.info(f"\n详细结果已保存到: {output_file}")
            logger.info(f"前5名相似文件对比图表已保存到: {os.path.join(output_dir, 'top_5_similarity_comparison.png')}")
            
            # 计算并记录总执行时间
            total_time = time.time() - total_start_time
            logger.info(f"\n总执行时间: {total_time:.2f}秒")
            logger.info("各阶段耗时统计:")
            logger.info(f"- 数据加载: {load_time:.2f}秒 ({load_time/total_time*100:.1f}%)")
            logger.info(f"- 特征标准化: {normalize_time:.2f}秒 ({normalize_time/total_time*100:.1f}%)")
            logger.info(f"- 最大距离计算: {max_dist_time:.2f}秒 ({max_dist_time/total_time*100:.1f}%)")
            logger.info(f"- 相似度计算: {similarity_time:.2f}秒 ({similarity_time/total_time*100:.1f}%)")
            logger.info(f"- 结果保存: {save_time:.2f}秒 ({save_time/total_time*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"计算过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 