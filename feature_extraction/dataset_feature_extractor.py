import os
import json
from typing import Dict, Any
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from feature_extractor import FeatureExtractor

def process_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    处理整个数据集目录
    :param dataset_path: 数据集根目录路径
    :return: 包含所有数据集特征的字典
    """
    all_features = {}
    
    # 遍历数据集目录
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.fjs'):
                # 获取相对路径作为键名，使用正斜杠替换反斜杠
                rel_path = os.path.relpath(os.path.join(root, file), dataset_path).replace('\\', '/')
                print(f"正在处理文件: {rel_path}")
                
                try:
                    parameters = parser.parse(os.path.join(root, file))
                    extractor = FeatureExtractor(parameters)
                    features = extractor.extract_all_features()
                    all_features[rel_path] = features
                    
                except Exception as e:
                    print(f"处理文件 {rel_path} 时发生错误: {str(e)}")
                    continue
    
    return all_features

def main():
    try:
        # 获取数据集目录路径
        dataset_path = os.path.join(project_root, "dataset")
        print("开始处理数据集...")
        all_features = process_dataset(dataset_path)
        
        # 创建output目录（如果不存在）
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征到JSON文件
        output_path = os.path.join(output_dir, "dataset_features.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=2, ensure_ascii=False)
        
        print(f"\n特征提取完成，共处理 {len(all_features)} 个fjs文件")
        print(f"特征数据保存到: {output_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 