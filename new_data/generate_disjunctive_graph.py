import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from initial_validation.utils import parser
from initial_validation.disjunctive_graph import DisjunctiveGraph

def main():
    try:
        # 输入文件路径
        input_file = os.path.join("new_data", "new_fjsp_data.fjs")
        
        # 输出目录
        output_dir = os.path.join("new_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # 输出文件路径
        output_file = os.path.join(output_dir, "new_fjsp_data_disjunctive_graph.png")
        
        print(f"正在处理文件: {input_file}")
        
        # 解析数据文件
        parameters = parser.parse(input_file)
        
        # 创建析取图
        dg = DisjunctiveGraph(parameters)
        
        # 设置图表标题
        title = "新生成的FJSP数据析取图"
        
        # 保存析取图
        dg.draw(save_path=output_file, title=title)
        print(f"析取图已保存到: {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 