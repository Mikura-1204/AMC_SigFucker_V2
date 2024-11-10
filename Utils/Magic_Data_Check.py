import numpy as np
import os


#####------------------------------------------------------------------------------------#####


def test_read_npy(file_path, num_elements_to_print=10):
    """
    读取指定的 .npy 文件，打印文件的维度，并打印部分内容。

    参数:
    - file_path: str, .npy 文件的路径
    - num_elements_to_print: int, 打印的元素数量，默认为 10
    """
    try:
        # 读取 .npy 文件
        data = np.load(file_path)
        
        # 打印文件的维度
        print(f"文件维度是: {data.shape} 喵")
        if len(data.shape) == 2:
            # 处理二维数组
            print("前十个值如下喵:")
            print(data[:num_elements_to_print])
        elif len(data.shape) == 3:
            # 处理三维数组
            first_sample = data[0, :, :]
            print("第一个通道的前十个值如下喵:")
            print(first_sample[0, :num_elements_to_print])
            print("第二个通道的前十个值如下喵:")
            print(first_sample[1, :num_elements_to_print])
        elif len(data.shape) == 4:
            # 处理四维数组
            first_sample = data[0, :, :, :]
            print("第一个通道的前十个值如下喵:")
            print(first_sample[0, :num_elements_to_print, :num_elements_to_print])
            print("第二个通道的前十个值如下喵:")
            print(first_sample[1, :num_elements_to_print, :num_elements_to_print])
        else:
            print("不支持的数组维度喵")
        
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到喵 ")
    except ValueError as ve:
        print(f"读取文件 {file_path} 时发生错误: {ve} 喵")
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e} 喵")
        

#####------------------------------------------------------------------------------------#####


def npy_value_check(file_path):
    """
    从 .npy 文件中读取非零值并打印出来。

    参数:
    file_path (str): .npy 文件的路径。
    """
    # 加载 .npy 文件
    arr = np.load(file_path)
    
    # 获取非零元素的索引
    nonzero_indices = np.nonzero(arr)
    
    # 使用索引提取非零值
    nonzero_values = arr[nonzero_indices]
    
    # 打印非零值
    print("非零值:")
    for value in nonzero_values:
        print(value)


#####------------------------------------------------------------------------------------#####