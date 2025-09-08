#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化验证流程图生成器
专门分析initialization_validity_verification_mp.py的代码逻辑并生成流程图
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class InitializationValidationFlowchartGenerator:
    """初始化验证流程图生成器"""
    
    def __init__(self):
        self.main_functions = []
        self.process_flows = []
        self.data_flows = []
        
    def analyze_code_structure(self) -> Dict:
        """分析代码结构"""
        code_structure = {
            'main_functions': [
                {
                    'name': 'setup_logging',
                    'description': '设置日志记录系统',
                    'inputs': ['log_dir'],
                    'outputs': ['logger'],
                    'purpose': '配置日志格式和输出'
                },
                {
                    'name': 'find_convergence_point',
                    'description': '使用Savitzky-Golay滤波和峰值检测找到收敛点',
                    'inputs': ['curve'],
                    'outputs': ['convergence_idx'],
                    'purpose': '分析收敛曲线确定收敛代数'
                },
                {
                    'name': 'plot_convergence_curves',
                    'description': '绘制收敛曲线图',
                    'inputs': ['convergence_data', 'save_path'],
                    'outputs': ['convergence_plot'],
                    'purpose': '可视化不同初始化方法的收敛性能'
                },
                {
                    'name': 'save_validation_results',
                    'description': '保存验证结果到JSON文件',
                    'inputs': ['results_data', 'output_path'],
                    'outputs': ['json_file'],
                    'purpose': '将验证结果持久化存储'
                },
                {
                    'name': 'get_all_fjs_files',
                    'description': '递归查找所有.fjs文件',
                    'inputs': ['dataset_root'],
                    'outputs': ['fjs_files_list'],
                    'purpose': '扫描数据集目录获取所有实例文件'
                },
                {
                    'name': 'process_single_file',
                    'description': '处理单个fjs文件的函数，用于多进程执行',
                    'inputs': ['fjs_path', 'dataset_root', 'output_root', 'runs', 'max_iterations', 'meta_heuristic', 'total_files'],
                    'outputs': ['validation_results'],
                    'purpose': '对单个实例执行三种初始化方法的验证'
                }
            ],
            'main_process': {
                'name': '初始化验证主流程',
                'description': '多进程验证不同初始化策略的性能',
                'steps': [
                    '设置日志记录系统',
                    '配置执行参数',
                    '扫描数据集文件',
                    '多进程并行处理',
                    '统计执行结果',
                    '输出最终报告'
                ]
            },
            'initialization_methods': [
                'heuristic - 启发式初始化',
                'mixed - 混合初始化', 
                'random - 随机初始化'
            ],
            'data_structures': [
                'parameters - FJS问题参数',
                'results - 各初始化方法的结果',
                'convergence_data - 收敛曲线数据',
                'convergence_times - 收敛时间数据',
                'validation_results - 验证结果字典'
            ]
        }
        
        return code_structure
    
    def generate_main_flowchart(self, code_structure: Dict) -> str:
        """生成主流程图"""
        mermaid_code = ["flowchart TD"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef startNode fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px",
            "    classDef configNode fill:#e3f2fd,stroke:#1565c0,stroke-width:2px",
            "    classDef processNode fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef decisionNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef parallelNode fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px",
            "    classDef outputNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
            "    classDef endNode fill:#ffebee,stroke:#c62828,stroke-width:2px"
        ])
        
        # 开始节点
        mermaid_code.append("    start([程序启动]):::startNode")
        
        # 配置阶段
        mermaid_code.extend([
            "    setup_log[设置日志记录系统]:::configNode",
            "    config_params[配置执行参数<br/>数据集路径、输出路径、<br/>执行次数、最大迭代数]:::configNode"
        ])
        
        # 文件扫描阶段
        mermaid_code.extend([
            "    scan_files[扫描数据集目录<br/>获取所有.fjs文件]:::processNode",
            "    check_files{检查文件数量}:::decisionNode"
        ])
        
        # 多进程处理阶段
        mermaid_code.extend([
            "    create_pool[创建多进程池]:::parallelNode",
            "    parallel_process[并行处理文件]:::parallelNode"
        ])
        
        # 单个文件处理流程
        mermaid_code.extend([
            "    parse_fjs[解析FJS文件<br/>获取问题参数]:::processNode",
            "    init_methods[执行三种初始化方法<br/>heuristic/mixed/random]:::processNode"
        ])
        
        # 每种初始化方法的处理
        mermaid_code.extend([
            "    run_ga[运行遗传算法<br/>获取makespan和收敛曲线]:::processNode",
            "    calc_stats[计算统计指标<br/>均值、标准差、收敛代数]:::processNode",
            "    save_results[保存验证结果<br/>JSON格式]:::outputNode"
        ])
        
        # 统计和结束
        mermaid_code.extend([
            "    collect_stats[收集所有处理结果]:::processNode",
            "    generate_report[生成最终统计报告]:::outputNode",
            "    end([程序结束]):::endNode"
        ])
        
        # 连接关系
        mermaid_code.extend([
            "    start --> setup_log",
            "    setup_log --> config_params",
            "    config_params --> scan_files",
            "    scan_files --> check_files",
            "    check_files -->|文件存在| create_pool",
            "    check_files -->|无文件| end",
            "    create_pool --> parallel_process",
            "    parallel_process --> parse_fjs",
            "    parse_fjs --> init_methods",
            "    init_methods --> run_ga",
            "    run_ga --> calc_stats",
            "    calc_stats --> save_results",
            "    save_results --> collect_stats",
            "    collect_stats --> generate_report",
            "    generate_report --> end"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_single_file_process_flowchart(self, code_structure: Dict) -> str:
        """生成单个文件处理流程图"""
        mermaid_code = ["flowchart TD"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef inputNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef processNode fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef decisionNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef loopNode fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px",
            "    classDef outputNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px"
        ])
        
        # 输入
        mermaid_code.extend([
            "    fjs_file[FJS文件]:::inputNode",
            "    check_exist{检查结果文件<br/>是否已存在?}:::decisionNode"
        ])
        
        # 文件处理
        mermaid_code.extend([
            "    parse_file[解析FJS文件<br/>获取机器数、作业数]:::processNode",
            "    init_results[初始化结果存储<br/>heuristic/mixed/random]:::processNode"
        ])
        
        # 三种初始化方法循环
        mermaid_code.extend([
            "    method_loop[遍历三种初始化方法<br/>heuristic → mixed → random]:::loopNode",
            "    current_method[当前初始化方法]:::processNode"
        ])
        
        # 多次运行循环
        mermaid_code.extend([
            "    run_loop[执行多次运行<br/>(默认20次)]:::loopNode",
            "    run_ga[运行遗传算法<br/>获取makespan和收敛曲线]:::processNode",
            "    find_convergence[计算收敛点<br/>Savitzky-Golay滤波]:::processNode",
            "    store_results[存储运行结果]:::processNode"
        ])
        
        # 统计计算
        mermaid_code.extend([
            "    calc_method_stats[计算方法统计指标<br/>均值、标准差、收敛代数]:::processNode",
            "    build_validation[构建验证结果字典]:::processNode",
            "    save_json[保存为JSON文件]:::outputNode"
        ])
        
        # 连接关系
        mermaid_code.extend([
            "    fjs_file --> check_exist",
            "    check_exist -->|不存在| parse_file",
            "    check_exist -->|已存在| skip[跳过处理]",
            "    parse_file --> init_results",
            "    init_results --> method_loop",
            "    method_loop --> current_method",
            "    current_method --> run_loop",
            "    run_loop --> run_ga",
            "    run_ga --> find_convergence",
            "    find_convergence --> store_results",
            "    store_results --> run_loop",
            "    run_loop --> calc_method_stats",
            "    calc_method_stats --> method_loop",
            "    method_loop --> build_validation",
            "    build_validation --> save_json"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_data_flow_diagram(self, code_structure: Dict) -> str:
        """生成数据流图"""
        mermaid_code = ["flowchart LR"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef dataNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
            "    classDef processNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef outputNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px"
        ])
        
        # 数据源
        mermaid_code.extend([
            "    dataset[dataset目录<br/>FJS文件]:::dataNode",
            "    config[配置参数<br/>runs, max_iterations]:::dataNode"
        ])
        
        # 处理节点
        mermaid_code.extend([
            "    parser[parser.parse<br/>解析FJS文件]:::processNode",
            "    ga_algorithm[ga_new<br/>遗传算法]:::processNode",
            "    convergence[find_convergence_point<br/>收敛点检测]:::processNode",
            "    statistics[统计计算<br/>均值、标准差]:::processNode"
        ])
        
        # 输出节点
        mermaid_code.extend([
            "    json_results[validation_results.json<br/>验证结果]:::outputNode",
            "    log_file[日志文件<br/>执行记录]:::outputNode"
        ])
        
        # 数据流连接
        mermaid_code.extend([
            "    dataset --> parser",
            "    config --> ga_algorithm",
            "    parser --> ga_algorithm",
            "    ga_algorithm --> convergence",
            "    convergence --> statistics",
            "    statistics --> json_results",
            "    ga_algorithm --> log_file"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_function_dependency_diagram(self, code_structure: Dict) -> str:
        """生成函数依赖关系图"""
        mermaid_code = ["graph TD"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef mainFunc fill:#ffebee,stroke:#c62828,stroke-width:3px",
            "    classDef utilFunc fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px",
            "    classDef dataFunc fill:#fff3e0,stroke:#ef6c00,stroke-width:2px"
        ])
        
        # 主函数
        mermaid_code.extend([
            "    main[__main__<br/>主程序入口]:::mainFunc"
        ])
        
        # 工具函数
        mermaid_code.extend([
            "    setup_logging[setup_logging<br/>日志设置]:::utilFunc",
            "    get_files[get_all_fjs_files<br/>文件扫描]:::utilFunc",
            "    save_results[save_validation_results<br/>结果保存]:::utilFunc"
        ])
        
        # 数据处理函数
        mermaid_code.extend([
            "    process_file[process_single_file<br/>单文件处理]:::dataFunc",
            "    find_conv[find_convergence_point<br/>收敛点检测]:::dataFunc",
            "    plot_curves[plot_convergence_curves<br/>曲线绘制]:::dataFunc"
        ])
        
        # 依赖关系
        mermaid_code.extend([
            "    main --> setup_logging",
            "    main --> get_files",
            "    main --> process_file",
            "    process_file --> find_conv",
            "    process_file --> save_results",
            "    process_file --> plot_curves"
        ])
        
        return "\n".join(mermaid_code)
    
    def generate_multiprocessing_flowchart(self, code_structure: Dict) -> str:
        """生成多进程处理流程图"""
        mermaid_code = ["flowchart TD"]
        
        # 定义节点样式
        mermaid_code.extend([
            "    classDef masterNode fill:#ffebee,stroke:#c62828,stroke-width:3px",
            "    classDef workerNode fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px",
            "    classDef dataNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px"
        ])
        
        # 主进程
        mermaid_code.extend([
            "    master[主进程<br/>Master Process]:::masterNode",
            "    file_list[文件列表<br/>FJS文件队列]:::dataNode"
        ])
        
        # 工作进程
        mermaid_code.extend([
            "    worker1[工作进程1<br/>Worker Process 1]:::workerNode",
            "    worker2[工作进程2<br/>Worker Process 2]:::workerNode",
            "    worker3[工作进程3<br/>Worker Process 3]:::workerNode",
            "    workerN[工作进程N<br/>Worker Process N]:::workerNode"
        ])
        
        # 处理流程
        mermaid_code.extend([
            "    task1[处理文件1<br/>解析→GA→统计]:::workerNode",
            "    task2[处理文件2<br/>解析→GA→统计]:::workerNode",
            "    task3[处理文件3<br/>解析→GA→统计]:::workerNode",
            "    taskN[处理文件N<br/>解析→GA→统计]:::workerNode"
        ])
        
        # 结果收集
        mermaid_code.extend([
            "    results[结果收集<br/>统计汇总]:::masterNode"
        ])
        
        # 连接关系
        mermaid_code.extend([
            "    master --> file_list",
            "    file_list --> worker1",
            "    file_list --> worker2",
            "    file_list --> worker3",
            "    file_list --> workerN",
            "    worker1 --> task1",
            "    worker2 --> task2",
            "    worker3 --> task3",
            "    workerN --> taskN",
            "    task1 --> results",
            "    task2 --> results",
            "    task3 --> results",
            "    taskN --> results"
        ])
        
        return "\n".join(mermaid_code)
    
    def save_all_diagrams(self, code_structure: Dict, output_dir: str):
        """保存所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成不同类型的图表
        diagrams = {
            'main_flow': self.generate_main_flowchart(code_structure),
            'single_file_process': self.generate_single_file_process_flowchart(code_structure),
            'data_flow': self.generate_data_flow_diagram(code_structure),
            'function_dependency': self.generate_function_dependency_diagram(code_structure),
            'multiprocessing': self.generate_multiprocessing_flowchart(code_structure)
        }
        
        # 保存Mermaid文件
        for diagram_type, mermaid_code in diagrams.items():
            mmd_file = os.path.join(output_dir, f"initialization_validation_{diagram_type}.mmd")
            html_file = os.path.join(output_dir, f"initialization_validation_{diagram_type}.html")
            
            with open(mmd_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            
            self._generate_html_preview(mermaid_code, html_file, f"初始化验证{diagram_type}图")
        
        # 生成PNG流程图
        self.generate_main_flowchart_png(code_structure, output_dir)
        self.generate_single_file_process_png(code_structure, output_dir)
        self.generate_multiprocessing_png(code_structure, output_dir)
        
        # 保存代码结构信息
        info_file = os.path.join(output_dir, "initialization_validation_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(code_structure, f, indent=2, ensure_ascii=False)
        
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
        .description {{
            background-color: #f3e5f5;
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
            <h3>文件信息</h3>
            <p><strong>文件名:</strong> initialization_validity_verification_mp.py</p>
            <p><strong>功能:</strong> 多进程验证不同初始化策略的性能</p>
            <p><strong>主要特点:</strong> 支持heuristic、mixed、random三种初始化方法</p>
        </div>
        <div class="description">
            <h3>核心功能</h3>
            <ul>
                <li>多进程并行处理FJS实例文件</li>
                <li>对每个实例执行三种初始化策略</li>
                <li>计算makespan统计指标和收敛性能</li>
                <li>生成验证结果JSON文件和收敛曲线图</li>
            </ul>
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
    
    def generate_main_flowchart_png(self, code_structure: Dict, output_dir: str):
        """生成主流程图的PNG版本"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # 定义节点位置和样式
        nodes = {
            'start': {'pos': (5, 11), 'text': '程序启动', 'color': '#c8e6c9', 'edgecolor': '#2e7d32'},
            'setup_log': {'pos': (5, 10), 'text': '设置日志记录系统', 'color': '#e3f2fd', 'edgecolor': '#1565c0'},
            'config_params': {'pos': (5, 9), 'text': '配置执行参数\n数据集路径、输出路径\n执行次数、最大迭代数', 'color': '#e3f2fd', 'edgecolor': '#1565c0'},
            'scan_files': {'pos': (5, 8), 'text': '扫描数据集目录\n获取所有.fjs文件', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'check_files': {'pos': (5, 7), 'text': '检查文件数量', 'color': '#fce4ec', 'edgecolor': '#880e4f'},
            'create_pool': {'pos': (5, 6), 'text': '创建多进程池', 'color': '#f3e5f5', 'edgecolor': '#6a1b9a'},
            'parallel_process': {'pos': (5, 5), 'text': '并行处理文件', 'color': '#f3e5f5', 'edgecolor': '#6a1b9a'},
            'parse_fjs': {'pos': (5, 4), 'text': '解析FJS文件\n获取问题参数', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'init_methods': {'pos': (5, 3), 'text': '执行三种初始化方法\nheuristic/mixed/random', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'run_ga': {'pos': (5, 2), 'text': '运行遗传算法\n获取makespan和收敛曲线', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'calc_stats': {'pos': (5, 1), 'text': '计算统计指标\n均值、标准差、收敛代数', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'save_results': {'pos': (5, 0.5), 'text': '保存验证结果\nJSON格式', 'color': '#e8f5e8', 'edgecolor': '#1b5e20'},
            'collect_stats': {'pos': (5, 0), 'text': '收集所有处理结果', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'generate_report': {'pos': (5, -0.5), 'text': '生成最终统计报告', 'color': '#e8f5e8', 'edgecolor': '#1b5e20'},
            'end': {'pos': (5, -1), 'text': '程序结束', 'color': '#ffebee', 'edgecolor': '#c62828'}
        }
        
        # 绘制节点
        for node_name, node_info in nodes.items():
            x, y = node_info['pos']
            text = node_info['text']
            color = node_info['color']
            edgecolor = node_info['edgecolor']
            
            # 根据节点类型选择形状
            if node_name in ['start', 'end']:
                # 椭圆节点
                ellipse = patches.Ellipse((x, y), 2, 0.8, color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(ellipse)
            elif node_name == 'check_files':
                # 菱形节点
                diamond = patches.Polygon([[x, y+0.4], [x+0.8, y], [x, y-0.4], [x-0.8, y]], 
                                        color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(diamond)
            else:
                # 矩形节点
                rect = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(rect)
            
            # 添加文本
            ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
        
        # 绘制连接线
        connections = [
            ('start', 'setup_log'),
            ('setup_log', 'config_params'),
            ('config_params', 'scan_files'),
            ('scan_files', 'check_files'),
            ('check_files', 'create_pool'),
            ('create_pool', 'parallel_process'),
            ('parallel_process', 'parse_fjs'),
            ('parse_fjs', 'init_methods'),
            ('init_methods', 'run_ga'),
            ('run_ga', 'calc_stats'),
            ('calc_stats', 'save_results'),
            ('save_results', 'collect_stats'),
            ('collect_stats', 'generate_report'),
            ('generate_report', 'end')
        ]
        
        for start_node, end_node in connections:
            start_pos = nodes[start_node]['pos']
            end_pos = nodes[end_node]['pos']
            
            # 绘制箭头
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # 添加标题
        ax.text(5, 11.5, '初始化验证程序主流程图', ha='center', va='center', 
               fontsize=16, weight='bold', color='#333')
        
        # 保存图片
        png_file = os.path.join(output_dir, "initialization_validation_main_flow.png")
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"主流程图PNG已保存: {png_file}")
    
    def generate_single_file_process_png(self, code_structure: Dict, output_dir: str):
        """生成单个文件处理流程图的PNG版本"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 定义节点位置和样式
        nodes = {
            'fjs_file': {'pos': (5, 9.5), 'text': 'FJS文件', 'color': '#e1f5fe', 'edgecolor': '#01579b'},
            'check_exist': {'pos': (5, 8.5), 'text': '检查结果文件\n是否已存在?', 'color': '#fce4ec', 'edgecolor': '#880e4f'},
            'parse_file': {'pos': (5, 7.5), 'text': '解析FJS文件\n获取机器数、作业数', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'init_results': {'pos': (5, 6.5), 'text': '初始化结果存储\nheuristic/mixed/random', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'method_loop': {'pos': (5, 5.5), 'text': '遍历三种初始化方法\nheuristic → mixed → random', 'color': '#f3e5f5', 'edgecolor': '#6a1b9a'},
            'current_method': {'pos': (5, 4.5), 'text': '当前初始化方法', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'run_loop': {'pos': (5, 3.5), 'text': '执行多次运行\n(默认20次)', 'color': '#f3e5f5', 'edgecolor': '#6a1b9a'},
            'run_ga': {'pos': (5, 2.5), 'text': '运行遗传算法\n获取makespan和收敛曲线', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'find_convergence': {'pos': (5, 1.5), 'text': '计算收敛点\nSavitzky-Golay滤波', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'store_results': {'pos': (5, 0.5), 'text': '存储运行结果', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'calc_method_stats': {'pos': (5, -0.5), 'text': '计算方法统计指标\n均值、标准差、收敛代数', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'build_validation': {'pos': (5, -1.5), 'text': '构建验证结果字典', 'color': '#fff3e0', 'edgecolor': '#e65100'},
            'save_json': {'pos': (5, -2.5), 'text': '保存为JSON文件', 'color': '#e8f5e8', 'edgecolor': '#1b5e20'},
            'skip': {'pos': (8, 8.5), 'text': '跳过处理', 'color': '#ffebee', 'edgecolor': '#c62828'}
        }
        
        # 绘制节点
        for node_name, node_info in nodes.items():
            x, y = node_info['pos']
            text = node_info['text']
            color = node_info['color']
            edgecolor = node_info['edgecolor']
            
            if node_name == 'check_exist':
                # 菱形节点
                diamond = patches.Polygon([[x, y+0.3], [x+0.6, y], [x, y-0.3], [x-0.6, y]], 
                                        color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(diamond)
            elif node_name in ['method_loop', 'run_loop']:
                # 循环节点（六边形）
                hexagon = patches.Polygon([[x-0.6, y], [x-0.3, y+0.3], [x+0.3, y+0.3], 
                                         [x+0.6, y], [x+0.3, y-0.3], [x-0.3, y-0.3]], 
                                        color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(hexagon)
            else:
                # 矩形节点
                rect = FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.6, 
                                    boxstyle="round,pad=0.1", 
                                    color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(rect)
            
            # 添加文本
            ax.text(x, y, text, ha='center', va='center', fontsize=8, weight='bold')
        
        # 绘制连接线
        connections = [
            ('fjs_file', 'check_exist'),
            ('check_exist', 'parse_file'),
            ('check_exist', 'skip'),
            ('parse_file', 'init_results'),
            ('init_results', 'method_loop'),
            ('method_loop', 'current_method'),
            ('current_method', 'run_loop'),
            ('run_loop', 'run_ga'),
            ('run_ga', 'find_convergence'),
            ('find_convergence', 'store_results'),
            ('store_results', 'run_loop'),
            ('run_loop', 'calc_method_stats'),
            ('calc_method_stats', 'method_loop'),
            ('method_loop', 'build_validation'),
            ('build_validation', 'save_json')
        ]
        
        for start_node, end_node in connections:
            start_pos = nodes[start_node]['pos']
            end_pos = nodes[end_node]['pos']
            
            # 绘制箭头
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # 添加标题
        ax.text(5, 9.8, '单个FJS文件处理流程图', ha='center', va='center', 
               fontsize=16, weight='bold', color='#333')
        
        # 保存图片
        png_file = os.path.join(output_dir, "initialization_validation_single_file_process.png")
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"单文件处理流程图PNG已保存: {png_file}")
    
    def generate_multiprocessing_png(self, code_structure: Dict, output_dir: str):
        """生成多进程处理流程图的PNG版本"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 定义节点位置和样式
        nodes = {
            'master': {'pos': (5, 9), 'text': '主进程\nMaster Process', 'color': '#ffebee', 'edgecolor': '#c62828'},
            'file_list': {'pos': (5, 7.5), 'text': '文件列表\nFJS文件队列', 'color': '#e1f5fe', 'edgecolor': '#01579b'},
            'worker1': {'pos': (2, 6), 'text': '工作进程1\nWorker Process 1', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'worker2': {'pos': (5, 6), 'text': '工作进程2\nWorker Process 2', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'worker3': {'pos': (8, 6), 'text': '工作进程3\nWorker Process 3', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'workerN': {'pos': (5, 5), 'text': '工作进程N\nWorker Process N', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'task1': {'pos': (2, 4), 'text': '处理文件1\n解析→GA→统计', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'task2': {'pos': (5, 4), 'text': '处理文件2\n解析→GA→统计', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'task3': {'pos': (8, 4), 'text': '处理文件3\n解析→GA→统计', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'taskN': {'pos': (5, 3), 'text': '处理文件N\n解析→GA→统计', 'color': '#e8f5e8', 'edgecolor': '#2e7d32'},
            'results': {'pos': (5, 1.5), 'text': '结果收集\n统计汇总', 'color': '#ffebee', 'edgecolor': '#c62828'}
        }
        
        # 绘制节点
        for node_name, node_info in nodes.items():
            x, y = node_info['pos']
            text = node_info['text']
            color = node_info['color']
            edgecolor = node_info['edgecolor']
            
            if node_name in ['master', 'results']:
                # 主进程节点（椭圆）
                ellipse = patches.Ellipse((x, y), 2.5, 0.8, color=color, edgecolor=edgecolor, linewidth=3)
                ax.add_patch(ellipse)
            else:
                # 工作进程节点（矩形）
                rect = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    color=color, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(rect)
            
            # 添加文本
            ax.text(x, y, text, ha='center', va='center', fontsize=8, weight='bold')
        
        # 绘制连接线
        connections = [
            ('master', 'file_list'),
            ('file_list', 'worker1'),
            ('file_list', 'worker2'),
            ('file_list', 'worker3'),
            ('file_list', 'workerN'),
            ('worker1', 'task1'),
            ('worker2', 'task2'),
            ('worker3', 'task3'),
            ('workerN', 'taskN'),
            ('task1', 'results'),
            ('task2', 'results'),
            ('task3', 'results'),
            ('taskN', 'results')
        ]
        
        for start_node, end_node in connections:
            start_pos = nodes[start_node]['pos']
            end_pos = nodes[end_node]['pos']
            
            # 绘制箭头
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # 添加标题
        ax.text(5, 9.5, '多进程并行处理流程图', ha='center', va='center', 
               fontsize=16, weight='bold', color='#333')
        
        # 保存图片
        png_file = os.path.join(output_dir, "initialization_validation_multiprocessing.png")
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"多进程处理流程图PNG已保存: {png_file}")


def main():
    """主函数"""
    # 初始化生成器
    generator = InitializationValidationFlowchartGenerator()
    
    print("正在分析初始化验证代码结构...")
    code_structure = generator.analyze_code_structure()
    
    print("正在生成流程图...")
    output_dir = "initialization_validation_flowcharts"
    generator.save_all_diagrams(code_structure, output_dir)
    
    print("\n生成完成!")
    print(f"输出目录: {output_dir}")
    print(f"主要函数数: {len(code_structure['main_functions'])}")
    print(f"初始化方法数: {len(code_structure['initialization_methods'])}")
    print(f"数据结构数: {len(code_structure['data_structures'])}")


if __name__ == "__main__":
    main() 