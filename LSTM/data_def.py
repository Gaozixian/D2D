import ast
import numpy as np

# 提供了一种办法将csv读取的str转换为list
# def str_to_list(s):
#     try:
#         # 用 ast.literal_eval 解析字符串列表
#         return ast.literal_eval(s)
#     except (ValueError, SyntaxError):
#         # 若解析失败，返回空列表或原数据（根据需求调整）
#         return []

def str_or_float_to_list(s):
    try:
        # 如果是字符串，用 ast.literal_eval 解析
        if isinstance(s, str):
            return ast.literal_eval(s)
        # 如果是数字，转换为单元素列表
        elif isinstance(s, (int, float, np.float64)):
            return [float(s)]
        # 如果已经是列表，直接返回
        elif isinstance(s, list):
            return s
        else:
            return []
    except (ValueError, SyntaxError):
        return []
