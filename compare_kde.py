import json

# 读取dataset_features.json中的KDE特征
with open('output/dataset_features.json', 'r', encoding='utf-8') as f:
    dataset_data = json.load(f)

# 读取原始的kde_results.json
with open('output/PDF_KDE_generator/kde_results.json', 'r') as f:
    kde_data = json.load(f)

# 获取样本数据
sample_key = 'Barnes/mt10c1.fjs'
dataset_kde = dataset_data[sample_key]['kde_features']
kde_sample_key = list(kde_data.keys())[0]
original_kde = kde_data[kde_sample_key]

print('dataset_features.json中的kde_features keys:')
print(list(dataset_kde.keys()))
print()

print('原始kde_results.json中的keys:')
print(list(original_kde.keys()))
print()

print('差异分析:')
dataset_keys = set(dataset_kde.keys())
original_keys = set(original_kde.keys())

extra_in_dataset = dataset_keys - original_keys
missing_in_dataset = original_keys - dataset_keys

if extra_in_dataset:
    print(f'在dataset_features.json中多出的keys: {list(extra_in_dataset)}')
if missing_in_dataset:
    print(f'在dataset_features.json中缺失的keys: {list(missing_in_dataset)}')
if not extra_in_dataset and not missing_in_dataset:
    print('两个文件的结构完全一致')

print('\nbandwidth值比较:')
print(f'dataset_features.json: {dataset_kde["bandwidth"]}')
print(f'kde_results.json: {original_kde["bandwidth"]}')
print(f'是否一致: {dataset_kde["bandwidth"] == original_kde["bandwidth"]}') 