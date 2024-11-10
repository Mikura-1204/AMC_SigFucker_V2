import numpy as np
import os


#####------------------------------------------------------------------------------------#####


def get_nonzero_values(arr):
    """
    返回 NumPy 数组中的非零值。

    参数:
    arr (numpy.ndarray): 输入的 NumPy 数组。

    返回:
    numpy.ndarray: 包含非零值的一维数组。
    """
    # 获取非零元素的索引
    nonzero_indices = np.nonzero(arr)
    
    # 使用索引提取非零值
    nonzero_values = arr[nonzero_indices]
    
    return nonzero_values


#####------------------------------------------------------------------------------------#####

