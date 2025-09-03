#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 修复 initialization_strategy_recommender_knn.py 文件中的错误

with open('initialization_strategy_recommender_knn.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 替换错误的行
found_fixes = 0
for i, line in enumerate(lines):
    if i >= 1320 and i <= 1350:  # 只检查问题区域
        print(f"检查第{i+1}行: {line.strip()}")
        if "stage_one_results" in line:
            print(f"找到问题行 {i+1}: {line.strip()}")
            if "stage_one_data = results['stage_one_results']['candidate_samples']" in line:
                lines[i] = "        knn_data = results['knn_search_results']['neighbors']\n"
                print(f"修复第{i+1}行: {line.strip()} -> {lines[i].strip()}")
                found_fixes += 1
            elif "candidate_samples = results['stage_one_results']['candidate_samples']" in line:
                lines[i] = "        # 已移除复杂的详细评分计算\n"
                print(f"修复第{i+1}行: {line.strip()} -> {lines[i].strip()}")
                found_fixes += 1
        elif "for i, data in enumerate(stage_one_data, 1):" in line:
            lines[i] = "        for i, data in enumerate(knn_data, 1):\n"
            print(f"修复第{i+1}行: {line.strip()} -> {lines[i].strip()}")
            found_fixes += 1

print(f"总共修复了 {found_fixes} 行")

# 保存修复后的文件
with open('initialization_strategy_recommender_knn.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("文件修复完成!")
