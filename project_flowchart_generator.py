#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目架构流程图生成器
专门为Dispatch_Sample_Generator项目生成整体架构流程图
"""

import os
import json
import ast
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

class ProjectFlowchartGenerator:
    """项目架构流程图生成器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.modules = {}
        self.data_flows = []
        self.dependencies = {}
        
    def analyze_project_structure(self) -> Dict:
        """分析项目整体结构"""
        project_info = {
            'name': 'Dispatch_Sample_Generator',
            'modules': {},
            'data_flows': [],
            'main_processes': []
        }
        
        # 分析主要模块
        main_modules = [
            'feature_extraction',
            'initial_validation', 
            'comparison_disjunctive_graphs',
            'comparison_graph_structure',
            'recommend_model_1',
            'recommend_model_2',
            'new_data',
            'fjs_generator',
            'PDF_KDE_generator',
            'feature_similarity_weighting'
        ]
        
        for module in main_modules:
            module_path = self.project_root / module
            if module_path.exists():
                module_info = self._analyze_module(module_path, module)
                project_info['modules'][module] = module_info
        
        # 分析数据流
        project_info['data_flows'] = self._analyze_data_flows()
        
        # 分析主要处理流程
        project_info['main_processes'] = self._analyze_main_processes()
        
        return project_info
    
    def _analyze_module(self, module_path: Path, module_name: str) -> Dict:
        """分析单个模块"""
        module_info = {
            'name': module_name,
            'files': [],
            'functions': [],
            'classes': [],
            'dependencies': [],
            'outputs': []
        }
        
        # 查找Python文件
        for py_file in module_path.rglob('*.py'):
            if py_file.is_file():
                file_info = self._analyze_python_file(py_file, module_name)
                module_info['files'].append(file_info)
                module_info['functions'].extend(file_info.get('functions', []))
                module_info['classes'].extend(file_info.get('classes', []))
        
        # 查找输出文件
        output_dirs = ['output', 'exp_result', 'result']
        for output_dir in output_dirs:
            output_path = self.project_root / output_dir
            if output_path.exists():
                for file_path in output_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix in ['.json', '.png', '.csv']:
                        module_info['outputs'].append(str(file_path.relative_to(self.project_root)))
        
        return module_info
    
    def _analyze_python_file(self, file_path: Path, module_name: str) -> Dict:
        """分析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_info = {
                'name': file_path.name,
                'path': str(file_path.relative_to(self.project_root)),
                'functions': [],
                'classes': [],
                'imports': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    }
                    file_info['functions'].append(func_info)
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [],
                        'docstring': ast.get_docstring(node)
                    }
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                    file_info['classes'].append(class_info)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    import_info = self._extract_import_info(node)
                    file_info['imports'].append(import_info)
            
            return file_info
            
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return {'name': file_path.name, 'path': str(file_path.relative_to(self.project_root))}
    
    def _extract_import_info(self, node) -> Dict:
        """提取导入信息"""
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'modules': [alias.name for alias in node.names]
            }
        else:  # ImportFrom
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [alias.name for alias in node.names]
            }
    
    def _analyze_data_flows(self) -> List[Dict]:
        """分析数据流"""
        data_flows = [
            {
                'from': 'dataset',
                'to': 'feature_extraction',
                'data_type': 'FJS数据文件',
                'description': '历史数据集输入'
            },
            {
                'from': 'feature_extraction',
                'to': 'output/dataset_features.json',
                'data_type': '特征数据',
                'description': '提取基础特征和加工时间特征'
            },
            {
                'from': 'PDF_KDE_generator',
                'to': 'output/PDF_KDE_generator',
                'data_type': 'KDE参数和PDF图',
                'description': '生成核密度估计'
            },
            {
                'from': 'initial_validation',
                'to': 'output/record_result',
                'data_type': '验证结果',
                'description': '不同初始化策略的性能验证'
            },
            {
                'from': 'comparison_disjunctive_graphs',
                'to': 'output/disjunctive_graphs',
                'data_type': '析取图',
                'description': '生成析取图结构'
            },
            {
                'from': 'new_data',
                'to': 'feature_similarity_weighting',
                'data_type': '新数据特征',
                'description': '生成新数据并提取特征'
            },
            {
                'from': 'feature_similarity_weighting',
                'to': 'recommend_model_1',
                'data_type': '相似度计算结果',
                'description': '计算新数据与历史数据的相似度'
            },
            {
                'from': 'recommend_model_1',
                'to': 'result/recommender_output',
                'data_type': '推荐结果',
                'description': '生成初始化策略推荐'
            }
        ]
        return data_flows
    
    def _analyze_main_processes(self) -> List[Dict]:
        """分析主要处理流程"""
        processes = [
            {
                'name': '特征提取流程',
                'steps': [
                    '解析FJS数据文件',
                    '提取基础特征（作业数、机器数等）',
                    '提取加工时间特征（均值、标准差等）',
                    '生成KDE概率密度估计',
                    '构建析取图结构'
                ]
            },
            {
                'name': '验证流程',
                'steps': [
                    '加载历史数据集',
                    '应用不同初始化策略',
                    '运行遗传算法',
                    '记录性能指标',
                    '生成收敛曲线'
                ]
            },
            {
                'name': '推荐流程',
                'steps': [
                    '提取新数据特征',
                    '计算与历史数据的相似度',
                    '选择最相似的候选样本',
                    '基于相似度加权推荐策略',
                    '生成推荐结果和可视化'
                ]
            }
        ]
        return processes
    
    def generate_project_architecture_diagram(self, project_info: Dict) -> str:
        """生成项目架构图"""
        mermaid_code = ["graph TB"]
        
        # 添加模块节点
        for module_name, module_info in project_info['modules'].items():
            node_id = f"module_{module_name}"
            label = f"{module_name}<br/>文件数: {len(module_info['files'])}<br/>函数数: {len(module_info['functions'])}"
            mermaid_code.append(f"    {node_id}[{label}]")
        
        # 添加数据流
        for flow in project_info['data_flows']:
            from_node = f"module_{flow['from']}" if flow['from'] in project_info['modules'] else flow['from']
            to_node = f"module_{flow['to']}" if flow['to'] in project_info['modules'] else flow['to']
            mermaid_code.append(f"    {from_node} -->|{flow['data_type']}| {to_node}")
        
        return "\n".join(mermaid_code)
    
    def generate_data_flow_diagram(self, project_info: Dict) -> str:
        """生成数据流图"""
        mermaid_code = ["flowchart LR"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef dataNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef processNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef outputNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px"
        ])
        
        # 数据源节点
        mermaid_code.extend([
            "    dataset[历史数据集<br/>dataset/]:::dataNode",
            "    newdata[新数据生成<br/>new_data/]:::dataNode"
        ])
        
        # 处理节点
        mermaid_code.extend([
            "    feature_ext[特征提取<br/>feature_extraction/]:::processNode",
            "    kde_gen[KDE生成<br/>PDF_KDE_generator/]:::processNode",
            "    validation[验证分析<br/>initial_validation/]:::processNode",
            "    comparison[图比较<br/>comparison_disjunctive_graphs/]:::processNode",
            "    similarity[相似度计算<br/>feature_similarity_weighting/]:::processNode",
            "    recommender[推荐系统<br/>recommend_model_1/]:::processNode"
        ])
        
        # 输出节点
        mermaid_code.extend([
            "    output[输出结果<br/>output/]:::outputNode",
            "    result[推荐结果<br/>result/]:::outputNode"
        ])
        
        # 数据流连接
        mermaid_code.extend([
            "    dataset --> feature_ext",
            "    dataset --> validation",
            "    dataset --> comparison",
            "    newdata --> similarity",
            "    feature_ext --> kde_gen",
            "    feature_ext --> output",
            "    kde_gen --> output",
            "    validation --> output",
            "    comparison --> output",
            "    similarity --> recommender",
            "    recommender --> result"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_process_flowchart(self, project_info: Dict) -> str:
        """生成处理流程图"""
        mermaid_code = ["flowchart TD"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef startNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px",
            "    classDef processNode fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef decisionNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef endNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px"
        ])
        
        # 开始节点
        mermaid_code.append("    start[开始]:::startNode")
        
        # 主要处理步骤
        mermaid_code.extend([
            "    load_data[加载历史数据集]:::processNode",
            "    extract_features[提取特征]:::processNode",
            "    generate_kde[生成KDE]:::processNode",
            "    validate[验证初始化策略]:::processNode",
            "    generate_new[生成新数据]:::processNode",
            "    calculate_sim[计算相似度]:::processNode",
            "    recommend[推荐策略]:::processNode",
            "    output_results[输出结果]:::processNode"
        ])
        
        # 决策节点
        mermaid_code.extend([
            "    check_data{数据是否有效?}:::decisionNode",
            "    check_similarity{相似度是否足够?}:::decisionNode"
        ])
        
        # 结束节点
        mermaid_code.append("    end[结束]:::endNode")
        
        # 连接关系
        mermaid_code.extend([
            "    start --> load_data",
            "    load_data --> check_data",
            "    check_data -->|是| extract_features",
            "    check_data -->|否| end",
            "    extract_features --> generate_kde",
            "    generate_kde --> validate",
            "    validate --> generate_new",
            "    generate_new --> calculate_sim",
            "    calculate_sim --> check_similarity",
            "    check_similarity -->|是| recommend",
            "    check_similarity -->|否| generate_new",
            "    recommend --> output_results",
            "    output_results --> end"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_module_dependency_diagram(self, project_info: Dict) -> str:
        """生成模块依赖图"""
        mermaid_code = ["graph LR"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef coreModule fill:#ffebee,stroke:#c62828,stroke-width:3px",
            "    classDef featureModule fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px",
            "    classDef analysisModule fill:#fff3e0,stroke:#ef6c00,stroke-width:2px",
            "    classDef outputModule fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px"
        ])
        
        # 核心模块
        mermaid_code.extend([
            "    parser[parser<br/>数据解析]:::coreModule",
            "    config[config<br/>配置管理]:::coreModule"
        ])
        
        # 特征模块
        mermaid_code.extend([
            "    feature_ext[feature_extraction<br/>特征提取]:::featureModule",
            "    kde_gen[PDF_KDE_generator<br/>KDE生成]:::featureModule",
            "    graph_gen[comparison_disjunctive_graphs<br/>析取图生成]:::featureModule"
        ])
        
        # 分析模块
        mermaid_code.extend([
            "    validation[initial_validation<br/>验证分析]:::analysisModule",
            "    similarity[feature_similarity_weighting<br/>相似度计算]:::analysisModule",
            "    recommender[recommend_model_1<br/>推荐系统]:::analysisModule"
        ])
        
        # 输出模块
        mermaid_code.extend([
            "    output[output<br/>结果输出]:::outputModule",
            "    result[result<br/>推荐结果]:::outputModule"
        ])
        
        # 依赖关系
        mermaid_code.extend([
            "    parser --> feature_ext",
            "    parser --> validation",
            "    parser --> graph_gen",
            "    config --> validation",
            "    config --> recommender",
            "    feature_ext --> kde_gen",
            "    feature_ext --> similarity",
            "    kde_gen --> similarity",
            "    graph_gen --> similarity",
            "    validation --> output",
            "    similarity --> recommender",
            "    recommender --> result"
        ])
        
        return "\n".join(mermaid_code)
    
    def save_diagrams(self, project_info: Dict, output_dir: str):
        """保存所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成不同类型的图表
        diagrams = {
            'architecture': self.generate_project_architecture_diagram(project_info),
            'data_flow': self.generate_data_flow_diagram(project_info),
            'process_flow': self.generate_process_flowchart(project_info),
            'module_dependency': self.generate_module_dependency_diagram(project_info)
        }
        
        # 保存Mermaid文件
        for diagram_type, mermaid_code in diagrams.items():
            mmd_file = os.path.join(output_dir, f"project_{diagram_type}.mmd")
            html_file = os.path.join(output_dir, f"project_{diagram_type}.html")
            
            with open(mmd_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            
            self._generate_html_preview(mermaid_code, html_file, f"项目{diagram_type}图")
        
        # 保存项目信息
        info_file = os.path.join(output_dir, "project_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(project_info, f, indent=2, ensure_ascii=False)
        
        print(f"所有图表已保存到: {output_dir}")
    
    def _generate_html_preview(self, mermaid_code: str, output_path: str, title: str):
        """生成HTML预览文件"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
        }}
        .info {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="info">
            <h3>项目信息</h3>
            <p>Dispatch_Sample_Generator - 调度样本生成器</p>
            <p>基于历史数据生成推荐模型所需的调度样本数据</p>
        </div>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }},
            graph: {{
                useMaxWidth: true,
                htmlLabels: true
            }}
        }});
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='项目架构流程图生成器')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--output-dir', default='project_flowcharts', help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = ProjectFlowchartGenerator(args.project_root)
    
    print("正在分析项目结构...")
    project_info = generator.analyze_project_structure()
    
    print("正在生成图表...")
    generator.save_diagrams(project_info, args.output_dir)
    
    print("\n生成完成!")
    print(f"输出目录: {args.output_dir}")
    print(f"项目模块数: {len(project_info['modules'])}")
    print(f"数据流数: {len(project_info['data_flows'])}")
    print(f"主要流程数: {len(project_info['main_processes'])}")


if __name__ == "__main__":
    main() 