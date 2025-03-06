# 用于解析fjs文件为机器和工件

from parse_fjs import parse

if __name__ == "__main__":
    # 标准解析fjsp数据集为json格式
    file_path = "dataset/Brandimarte/Text/Mk01.fjs"
    parsed_data =  parse(file_path)
    print(parsed_data)
