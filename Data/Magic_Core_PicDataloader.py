#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Magic Core is used to construct dataloader for Pic data. Just Processing Pic dataset.
#   Pack the Pic data for your Net.
#   [This Fucking File still need to complete]
#####------------------------------------------------------------------------------------#####

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.signal as scisig
from tqdm import tqdm
import re
from PIL import Image

class Pic_DataProcess():
    def __init__(self):
        self.ori_data_root = "/media/star/Elements/36s/dataset/type4(fs=320k)"
        self.processed_data_root = "/media/star/Elements/Ciallo_SigFucker_NewType4/Fucking_Signal_Pic"
        self.attributes = {}
        self.__check_dataset()
        self.info = {"class names": self.attributes.keys()}    # ciallo: what you should classify is the keys that filter
    
    # ciallo: after that the Dict self.attribute will record every class and corresponding file name
    #         if FILE CONSTRCTION change, CHECK THIS
    def __check_dataset(self):
        if not os.path.exists(self.ori_data_root):
            raise ValueError("根目录不存在喵")

        # 获取所有子目录名
        folder_name_list = [f for f in os.listdir(self.processed_data_root) if os.path.isdir(os.path.join(self.processed_data_root, f))]
        folder_name_list = sorted(folder_name_list)

        
        # 初始化属性字典
        self.attributes = {}

        for folder_name in folder_name_list:
            # 获取每个子目录下的所有 .jpg 文件
            jpg_file_list = [os.path.join(self.processed_data_root, folder_name, f) for f in os.listdir(os.path.join(self.processed_data_root, folder_name)) if f.endswith('.jpg') or f.endswith('.JPG')]
            
            # 对文件列表进行排序
            jpg_file_list = sorted(jpg_file_list)

            # 将该子目录下的文件列表存入字典，键为子目录名
            self.attributes[folder_name] = jpg_file_list
            
        # ciallo: add pic load method
    def load_image(self, image_path):
        # 确保文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        # 加载图像
        image = Image.open(image_path)
        image = image.resize(size=(64,64))
        
        # 将图像转换为 NumPy 数组
        image_array = np.array(image)
        
        image_array = image_array.transpose((2, 0, 1))
        
        return image_array
    
    # ciallo: change to load pic
    def load_data(self, class_name):
    # 只在这里构造一次 Burst 前缀
    
        class_name_num = class_name.split('t')[-1]
        sub_class_name = f"Burst{class_name_num}"  # 假设 class_name 是 1、2、3 等数字
        data_folder_path = os.path.join(self.processed_data_root, sub_class_name)

        # 检查数据文件夹是否存在
        if not os.path.exists(data_folder_path):
            raise FileNotFoundError(f"Data folder does not exist: {data_folder_path}")

        # 获取所有 jpg 文件，并过滤出符合条件的文件
        jpg_files = sorted([f for f in os.listdir(data_folder_path) if f.startswith("burst_") and f.endswith('.jpg')])
        
        signals = []
        
        # 读取每个符合条件的 jpg 文件
        for jpg_file in jpg_files:
            data_file_path = os.path.join(data_folder_path, jpg_file)
            
            signal = self.load_image(data_file_path)
            signals.append(signal)
        # return signals
        # 将 signals 列表转换为 NumPy 数组
        return np.array(signals)  # 返回 NumPy 数组