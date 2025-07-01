import json
import os

# 检查new_data_f目录中的特征
data_file = 'new_data_f/new_data_features.json'
if os.path.exists(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fjs_file = list(data.keys())[0]
    features = data[fjs_file]
    
    print('FJS文件:', fjs_file)
    print('特征字段:', list(features.keys()))
    print()
    
    print('basic_features keys:', list(features['basic_features'].keys()))
    print('processing_time_features keys:', list(features['processing_time_features'].keys()))
    print('kde_features keys:', list(features['kde_features'].keys()))
    print('disjunctive_graphs_features keys:', list(features['disjunctive_graphs_features'].keys()))
    
    print('\n验证特征结构完整性:')
    expected_features = ['basic_features', 'processing_time_features', 'kde_features', 'disjunctive_graphs_features']
    for feature in expected_features:
        if feature in features:
            print(f'✓ {feature}: 存在')
        else:
            print(f'✗ {feature}: 缺失')
else:
    print(f"文件 {data_file} 不存在") 