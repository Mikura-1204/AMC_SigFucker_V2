#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Magic Core is used to process Sig Data. Use it as main to cut the Sig for you need.
#   Especially for std files.
#####------------------------------------------------------------------------------------#####

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.signal as scisig
from tqdm import tqdm

class DataProcess():
    def __init__(self):
        self.ori_data_root = "/media/ubuntu/Elements/36s/dataset/type5(fs=30720k)"
        self.processed_data_root = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_CutNpy2"
        
        self.attributes = {}                                
        self.__construct_dataset_path()
        
        # ciallo: Use another Dict to restore all the class name (Back with all the key value, just Class Names)
        self.info = {"class names": self.attributes.keys()}        
    
    # ciallo: After that the Dict self.attribute will record every class and corresponding file name
    #         This method is used to get raw dataset which will be processed next
    def __construct_dataset_path(self):
        if not os.path.exists(self.ori_data_root):
            raise ValueError("根目录不存在喵 !")

        # ciallo: get sub Dict name as Class Name
        sub_folder_name_list = [f for f in os.listdir(self.ori_data_root) if os.path.isdir(os.path.join(self.ori_data_root, f))]
    
        for sub_folder_name in sub_folder_name_list:
            
            # ciallo GPT: 在'属性'的字典中,把每个类别当作一个键,这个键的内容再赋给一个空字典
            self.attributes[sub_folder_name] = {}
            sub_folder_path = os.path.join(self.ori_data_root, sub_folder_name,'burst')
        
            # 找到当前sub_folder_path内的所有.std文件 (可以修改成需要处理的文件名喵!)
            file_list = [os.path.join(sub_folder_path, f) for f in os.listdir(sub_folder_path) if f.endswith('.std') or f.endswith('.STD')]

            # ciallo GPT: 这个文件名键的空字典,其中一个键定义为std file,再赋给他文件列表作为值
            self.attributes[sub_folder_name]["std_files"] = file_list  # 使用一个键保存文件列表
    
    # ciallo: load data depending on different Sample-Freq in File-Name and change dual channels into two single channal with one offset
    def __load_data(self, data_path, dual):
        
        # ciallo GPT: 从二进制文件中读取数据,并将其转换为 NumPy 数组,读取的具体参数随你的文件更改喵
        signal = np.fromfile(data_path, dtype=np.dtype('<h'), count=-1, sep='', offset=50)
        if dual:
            signal = signal.reshape(-1, 2)    # line 1 is channel 1, another is channel 2
        else:
            signal = signal[::2]    # ignore one empty channel sig and select single message
            
        # ciallo: return a np-array
        return signal
    
    # ciallo: Use to check quality and Special-Charac  (if change the check, modify here)
    def __signal_check(self, signal, dual, mode=None):
        if dual:
            signal = signal.T

        # ciallo: but now no checking
        if mode == None:
            judge = True
            
        # ciallo: checking method is using STFT to check Power-Spectrum below the threshold with Var 
        elif mode == 'stft-variance':
            f, t, Sxx = scisig.stft(signal, window='hamming', nperseg=512)
            Sxx = 20 * np.log10(np.abs(Sxx) ** 2)
            threshold = 160
            variance = np.var(Sxx)
            plt.subplot(122)
            # plt.pcolormesh(t, f, Sxx)
            # plt.title("threshold={}  variance={:.2f}".format(threshold, variance))
            judge = variance > threshold
        # plt.show()
        # plt.close()
        return judge
    
    # ciallo: Cut with definding Step-Length of sample-length rate
    # ciallo GPT: num_sample是你要切的样本数喵,是几就切多少下喵; sample_length是单个切割窗的长度喵,
    #             overlap_rate是重叠率喵,如果是1.0就不重叠喵
    def cut_by_window(self, num_sample, sample_length = 65535, dual = True, overlap_rate = 1.0):
        print(" 开始切割喵 (∠・ω< )⌒☆ ", end='\n')
        os.makedirs(self.processed_data_root, exist_ok=True)
        used_file = {}
        for class_name in self.info["class names"]:
            
            # ciallo GPT: used_file把每类已经使用的文件名记录在此处喵
            #             in_class_names是每一类待处理的文件喵
            #             num_save_sample是已经处理的文件数量喵
            #             save_list是保存的切分完的数据喵
            used_file[class_name] = []                  
            in_class_names = self.attributes[class_name].keys()
            num_save_sample = 0
            save_list = []
                                      
            with tqdm(total = num_sample) as pbar:
                pbar.set_description("Processing {} nya ".format(class_name))
                for in_class_name in in_class_names:
                    file_path_list = self.attributes[class_name][in_class_name]
                    for file_path in file_path_list:
                        signal = self.__load_data(file_path, dual=dual)
                        used_file[class_name].append(file_path)
                        len_signal = len(signal)
                        cut_index = 0
                        
                        # ciallo: this is truly cutting progress
                        # ciallo GPT: 这里切割时,双通道一行是一组,所以是按行切的喵,切完为了处理方便转置了喵
                        while num_save_sample < num_sample and cut_index + sample_length < len_signal:
                            if dual:
                                single_sample = signal[cut_index:cut_index + sample_length, :]
                                single_sample = single_sample.T
                            else:
                                single_sample = signal[cut_index:cut_index + sample_length]
                                
                            # ciallo GPT: 对切割完的数据进行检查
                            if self.__signal_check(single_sample, dual = dual):
                                save_list.append(single_sample)
                                num_save_sample += 1
                                pbar.update(1)
                            cut_index += int(sample_length * overlap_rate)
                        if num_save_sample >= num_sample:
                            break
                    if num_save_sample >= num_sample:
                        break
            if num_save_sample < num_sample:
                print("[warning]: {}  样本数据不足喵.".format(class_name))
            
            # ciallo GPT: 每一类文件切割拼合到一个文件中
            np.save(os.path.join(self.processed_data_root, "{}.npy".format(class_name)), save_list)
        print("使用到的信号文件喵: ")
        print(used_file, end='\n')
    
    # ciallo: This Method is used to load cutted npy file (one for one class)
    def load_npy_data(self, class_name):
        data_file_name = "{}.npy".format(class_name)
        data_file_path = os.path.join(self.processed_data_root, data_file_name)
        signal = np.load(data_file_path)
        return signal
    
# ciallo GPT: 在上述切割完成后,在此处加载切割完的npy喵
def load_processed_dataset():
    data = []
    label = []
    dataset = DataProcess()
    class_names = dataset.info["class names"]    # 获取所有类别的名字，是一个字符串列表。每个类别都会有一组对应的数据文件
    
    # ciallo GPT: index 是类别的索引（从0开始），name 是类别名 index 是给每个类别名一个数字编码
    #             所以这个label其实就是全是同一个int的数组
    for index, name in enumerate(class_names):
        one_file_data = dataset.load_npy_data(class_name=name)
        one_file_label = np.ones(len(one_file_data), dtype=np.int64) * index
        
        # ciallo: concentrate data and label to a whole file
        data.extend(one_file_data)
        label.extend(one_file_label)
    data = np.array(data)
    label = np.array(label)
    
    return data, label, class_names
    
if __name__ == "__main__":
    dataset = DataProcess()
    
    # ciallo GPT: 切割函数执行之后, 输出的npy文件维度是[num_sample, channel, sample_length] 
    #             样本数,通道数,采样点数--采样长度
    dataset.cut_by_window(2000, sample_length=2048, dual=True, overlap_rate=1.0)
