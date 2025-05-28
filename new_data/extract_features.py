import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from feature_extraction.feature_extractor import FeatureExtractor

def main():
    try:
        # 输入文件路径
        input_file = os.path.join("new_data", "new_fjsp_data.fjs")
        
        # 输出文件路径
        output_file = os.path.join("new_data", "new_fjsp_features.json")
        
        print(f"正在处理文件: {input_file}")
        
        # 解析数据文件
        parameters = parser.parse(input_file)
        
        # 创建特征提取器
        feature_extractor = FeatureExtractor(parameters)
        
        # 提取特征
        features = feature_extractor.extract_all_features()
        
        # 创建特征字典
        feature_dict = {
            "new_fjsp_data": features
        }
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(feature_dict, f, indent=4, ensure_ascii=False)
        
        print(f"特征已保存到: {output_file}")
        print("\n提取的特征:")
        print(json.dumps(features, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 