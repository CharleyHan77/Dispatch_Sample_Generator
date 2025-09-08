#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码流程图生成器
基于Python代码逻辑自动生成Mermaid格式的流程图
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Optional
import json

class CodeFlowchartGenerator:
    """代码流程图生成器"""
    
    def __init__(self):
        self.functions = {}
        self.classes = {}
        self.flow_connections = []
        self.imports = []
        self.variables = {}
        
    def parse_python_file(self, file_path: str) -> Dict:
        """解析Python文件并提取代码结构"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            return self._analyze_ast(tree, file_path)
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return {}
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> Dict:
        """分析AST树并提取代码结构"""
        result = {
            'file_path': file_path,
            'functions': [],
            'classes': [],
            'imports': [],
            'main_flow': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._extract_function_info(node)
                result['functions'].append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node)
                result['classes'].append(class_info)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                import_info = self._extract_import_info(node)
                result['imports'].append(import_info)
            elif isinstance(node, ast.If):
                if_info = self._extract_conditional_info(node)
                result['main_flow'].append(if_info)
            elif isinstance(node, ast.For):
                for_info = self._extract_loop_info(node)
                result['main_flow'].append(for_info)
            elif isinstance(node, ast.While):
                while_info = self._extract_while_info(node)
                result['main_flow'].append(while_info)
        
        return result
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict:
        """提取函数信息"""
        return {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'line_number': node.lineno,
            'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node)),
            'calls': self._extract_function_calls(node)
        }
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict:
        """提取类信息"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._extract_function_info(item))
        
        return {
            'name': node.name,
            'bases': [self._get_base_name(base) for base in node.bases],
            'methods': methods,
            'docstring': ast.get_docstring(node),
            'line_number': node.lineno
        }
    
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
    
    def _extract_conditional_info(self, node: ast.If) -> Dict:
        """提取条件语句信息"""
        return {
            'type': 'if',
            'condition': self._get_condition_string(node.test),
            'line_number': node.lineno,
            'has_else': len(node.orelse) > 0
        }
    
    def _extract_loop_info(self, node: ast.For) -> Dict:
        """提取循环信息"""
        return {
            'type': 'for',
            'target': self._get_target_string(node.target),
            'iter': self._get_iter_string(node.iter),
            'line_number': node.lineno
        }
    
    def _extract_while_info(self, node: ast.While) -> Dict:
        """提取while循环信息"""
        return {
            'type': 'while',
            'condition': self._get_condition_string(node.test),
            'line_number': node.lineno
        }
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """提取函数调用"""
        calls = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    calls.append(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    calls.append(f"{self._get_attribute_string(n.func)}")
        return calls
    
    def _get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return "unknown_decorator"
    
    def _get_base_name(self, base) -> str:
        """获取基类名称"""
        if isinstance(base, ast.Name):
            return base.id
        return "unknown_base"
    
    def _get_condition_string(self, test) -> str:
        """获取条件字符串"""
        try:
            return ast.unparse(test)
        except:
            return "condition"
    
    def _get_target_string(self, target) -> str:
        """获取目标字符串"""
        try:
            return ast.unparse(target)
        except:
            return "target"
    
    def _get_iter_string(self, iter_node) -> str:
        """获取迭代器字符串"""
        try:
            return ast.unparse(iter_node)
        except:
            return "iter"
    
    def _get_attribute_string(self, attr) -> str:
        """获取属性字符串"""
        try:
            return ast.unparse(attr)
        except:
            return "attribute"
    
    def generate_mermaid_flowchart(self, code_structure: Dict) -> str:
        """生成Mermaid格式的流程图"""
        mermaid_code = ["flowchart TD"]
        
        # 添加函数节点
        for func in code_structure['functions']:
            node_id = f"func_{func['name']}"
            label = f"{func['name']}({', '.join(func['args'])})"
            mermaid_code.append(f"    {node_id}[{label}]")
        
        # 添加类节点
        for cls in code_structure['classes']:
            node_id = f"class_{cls['name']}"
            label = f"class {cls['name']}"
            mermaid_code.append(f"    {node_id}[{label}]")
            
            # 添加类的方法
            for method in cls['methods']:
                method_id = f"method_{cls['name']}_{method['name']}"
                method_label = f"{method['name']}({', '.join(method['args'])})"
                mermaid_code.append(f"    {method_id}[{method_label}]")
                mermaid_code.append(f"    {node_id} --> {method_id}")
        
        # 添加主流程节点
        for i, flow in enumerate(code_structure['main_flow']):
            node_id = f"flow_{i}"
            if flow['type'] == 'if':
                label = f"if {flow['condition']}"
                mermaid_code.append(f"    {node_id}{{{label}}}")
            elif flow['type'] == 'for':
                label = f"for {flow['target']} in {flow['iter']}"
                mermaid_code.append(f"    {node_id}[{label}]")
            elif flow['type'] == 'while':
                label = f"while {flow['condition']}"
                mermaid_code.append(f"    {node_id}[{label}]")
        
        # 添加连接关系
        for func in code_structure['functions']:
            for call in func['calls']:
                # 查找被调用的函数
                for other_func in code_structure['functions']:
                    if other_func['name'] == call:
                        mermaid_code.append(f"    func_{func['name']} --> func_{call}")
                        break
        
        return "\n".join(mermaid_code)
    
    def generate_class_diagram(self, code_structure: Dict) -> str:
        """生成类图"""
        mermaid_code = ["classDiagram"]
        
        for cls in code_structure['classes']:
            mermaid_code.append(f"    class {cls['name']} {{")
            
            # 添加方法
            for method in cls['methods']:
                args_str = ", ".join(method['args'])
                mermaid_code.append(f"        +{method['name']}({args_str})")
            
            mermaid_code.append("    }")
            
            # 添加继承关系
            for base in cls['bases']:
                if base != "unknown_base":
                    mermaid_code.append(f"    {base} <|-- {cls['name']}")
        
        return "\n".join(mermaid_code)
    
    def generate_sequence_diagram(self, code_structure: Dict, main_function: str = "main") -> str:
        """生成时序图"""
        mermaid_code = ["sequenceDiagram"]
        
        # 查找主函数
        main_func = None
        for func in code_structure['functions']:
            if func['name'] == main_function:
                main_func = func
                break
        
        if main_func:
            mermaid_code.append(f"    participant {main_function}")
            
            # 添加函数调用序列
            for call in main_func['calls']:
                mermaid_code.append(f"    {main_function}->>{call}: call")
                mermaid_code.append(f"    {call}-->>{main_function}: return")
        
        return "\n".join(mermaid_code)
    
    def save_mermaid_file(self, mermaid_code: str, output_path: str):
        """保存Mermaid代码到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            print(f"Mermaid代码已保存到: {output_path}")
        except Exception as e:
            print(f"保存文件失败: {e}")
    
    def generate_html_preview(self, mermaid_code: str, output_path: str):
        """生成HTML预览文件"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>代码流程图</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .mermaid {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>代码流程图</h1>
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
                htmlLabels: true
            }}
        }});
    </script>
</body>
</html>
"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            print(f"HTML预览文件已保存到: {output_path}")
        except Exception as e:
            print(f"保存HTML文件失败: {e}")


def main():
    """主函数：命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='代码流程图生成器')
    parser.add_argument('input_file', help='输入的Python文件路径')
    parser.add_argument('--output-dir', default='flowchart_output', help='输出目录')
    parser.add_argument('--diagram-type', choices=['flowchart', 'class', 'sequence'], 
                       default='flowchart', help='图表类型')
    parser.add_argument('--main-function', default='main', help='时序图的主函数名')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化生成器
    generator = CodeFlowchartGenerator()
    
    # 解析代码
    print(f"正在解析文件: {args.input_file}")
    code_structure = generator.parse_python_file(args.input_file)
    
    if not code_structure:
        print("解析失败，无法生成流程图")
        return
    
    # 生成Mermaid代码
    if args.diagram_type == 'flowchart':
        mermaid_code = generator.generate_mermaid_flowchart(code_structure)
    elif args.diagram_type == 'class':
        mermaid_code = generator.generate_class_diagram(code_structure)
    elif args.diagram_type == 'sequence':
        mermaid_code = generator.generate_sequence_diagram(code_structure, args.main_function)
    
    # 保存文件
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    mermaid_file = os.path.join(args.output_dir, f"{base_name}_{args.diagram_type}.mmd")
    html_file = os.path.join(args.output_dir, f"{base_name}_{args.diagram_type}.html")
    
    generator.save_mermaid_file(mermaid_code, mermaid_file)
    generator.generate_html_preview(mermaid_code, html_file)
    
    print(f"\n生成完成!")
    print(f"Mermaid文件: {mermaid_file}")
    print(f"HTML预览: {html_file}")
    print(f"\nMermaid代码预览:")
    print("=" * 50)
    print(mermaid_code)
    print("=" * 50)


if __name__ == "__main__":
    main() 