#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Magic Core is used to construct dataloader for npy data. Just Processing Sig dataset.
#   Pack the Sig data for your Net.
#####------------------------------------------------------------------------------------#####

import numpy as np
import pywt
from torch.utils.data import Dataset

# ciallo: While using, make sure that data-process and dataloader is in same dict called dataset
from Data import Magic_Core_SigDatasets

# ciallo: General Dataset class model    数据集加载、拆分、类型指定
class Dataset_Sig(Dataset):
    def __init__(self, dataset_name:str, data_split:list = [0.7, 0.1, 0.2], data_type:str = 'None'):
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.class_names = None
        self.data_type = data_type
        
    def __data_split(self, input_data, label, ratio):
        input_data = np.array(input_data)
        label = np.array(label)

        # ciallo: Use the first Dim of loaded data as Sample-Num
        n_examples = input_data.shape[0]

        # ciallo: X1 means Train-Set num. Random choose x1_index as train data.
        n_x1 = int(ratio[0] * n_examples)  
        x1_index = np.random.choice(range(0, n_examples), size=n_x1, replace=False)

        # ciallo GPT: 此处用减的方法validation_index并没有打乱
        x2_index = list(set(range(0, n_examples)) - set(x1_index))

        # ciallo: 1 means Train-Dataset. 2 means Test-Dataset (Split Data)
        x1 = input_data[x1_index]
        x2 = input_data[x2_index]
        y1 = label[x1_index]
        y2 = label[x2_index]
        
        return x1, y1, x2, y2
    
    # ciallo GPT: 根据划分的比例封装好训练、验证、测试集
    def get_dataset(self, test_source_flag=False):

        # ciallo: Load all process data
        data, label, self.class_names = Magic_Core_SigDatasets.load_processed_dataset()

        # ciallo: Split Dataset (Test incldues two part)
        train_data, train_label, other_data, other_label = self.__data_split(data, label, [self.data_split[0], sum(self.data_split[1:])])
        train_dataset = MyDataset(train_data, train_label, self.data_type)

        # ciallo GPT: test_source_flag 决定是否使用验证集 (影响训练方式)  同时划分验证和测试集
        if self.data_split[-1] != 0 and not test_source_flag:
            validation_data, validation_label, test_data, test_label = self.__data_split(other_data, other_label, [x / sum(self.data_split[1:]) for x in self.data_split[1:]])
            validation_dataset = MyDataset(validation_data, validation_label, self.data_type)
            test_dataset = MyDataset(test_data, test_label, self.data_type)
        else:
            validation_dataset = MyDataset(other_data, other_label, self.data_type)
            if test_source_flag:
                test_data = None
                test_dataset = MyDataset(test_data, test_label, self.data_type)
            else:
                test_dataset = MyDataset([], [], self.data_type)
        return train_dataset, validation_dataset, test_dataset, self.class_names
        
# ciallo GPT: 单类数据集封装
class MyDataset(Dataset):
    def __init__(self, data, label, data_type):
        self.data = np.array(data, dtype=np.float32)
        self.label = np.array(label, dtype=np.int64)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
    
    def __normlization(self, norm_type:str = 'None'):
        if norm_type == "average":
            self.data = normalization_by_average(self.data)
    
    # ciallo: If you want to do Trans to Sig-Data, add here.
    def __trans(self, trans_name:str = 'None'):
        if trans_name == "dwt":
            self.data = dwt_in_matrix(self.data)

# ciallo GPT: 类似一种居中归一化的方法
def normalization_by_average(datas):
    result = []
    for data in datas:
        data = data - np.average(data)
        data = data / np.max(np.abs(data))
        result.append(data)
    return np.array(result, dtype=np.float32)

# ciallo_Fuc: 离散小波变换函数 返回结果矩阵
def dwt_in_matrix(datas):
    wavelet = pywt.Wavelet('db1')    # 选择 Daubechies 小波基函数 db1
    outputs = []
    print(f"输入数据的维度如下喵: {datas.shape} ")
    level = int(np.log2(datas.shape[0]))    # 获取输入数据的最后一维长度，即每个信号的样本数
    for data in datas:
        i_dwt = pywt.wavedec(data[0], wavelet, level=level)
        q_dwt = pywt.wavedec(data[1], wavelet, level=level)
        matrix = np.zeros((2, level + 1, datas.shape[-1]))
        in_len = 0
        for i in range(level + 1):
            matrix[0, i, in_len:in_len+len(i_dwt[i])] = i_dwt[i]
            matrix[1, i, in_len:in_len+len(q_dwt[i])] = q_dwt[i]
            in_len += len(i_dwt[i])
        outputs.append(matrix)
    return np.array(outputs, dtype=np.float32)

# ciallo Fuc: a special Processing Method like below:
# 信号数据映射到图结构 输入-信号数据，通常是一个二维数组，每一行表示一个节点（或信号源），每一列表示一个时间步长或特征。
# 输出-图数据和图边关系矩阵。矩阵中的每个元素 A[i][j] 表示节点 i 和节点 j 之间的连接权重。如果节点 i 和节点 j 之间没有连接，通常 A[i][j] = 0
def limited_fixed_graph_mapping(datas, k, weight = False):
    edge = np.zeros(shape=(datas.shape[-1], datas.shape[-1]), dtype=np.float32)
    for i in range(datas.shape[-1]):
            for j in range(i, datas.shape[-1]):
                if j - i <= k and j - i != 0:
                    edge[i][j] = 1 / ((j - i) * int(weight) + int(not weight))
                elif j - i > k:
                    break
    edge = edge + edge.T
    datas = datas.transpose((0, 2, 1))

    return np.array(datas, dtype=np.float32), np.array(edge, dtype=np.float32)