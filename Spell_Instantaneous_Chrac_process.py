import os
import numpy as np

from Utils.Magic_Instantaneous_Charac import extract_instantaneous_features, plot_feature_heatmaps

raw_data_root = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_CutNpy2"
npy_output_root = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_Instantaneous_npy"
pic_output_root = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_Instantaneous_pic"

data_file_list = []
data_file_list = os.listdir(raw_data_root)

for file in data_file_list:
    file_basename, _ = os.path.splitext(file)
    each_file_path = os.path.join(raw_data_root, file)
    data = np.load(each_file_path)  # 文件路径
    amplitude, phase, frequency = extract_instantaneous_features(data)
    output_path_1 = os.path.join(npy_output_root, '{}_instantaneous_amplitude.npy'.format(file_basename))
    output_path_2 = os.path.join(npy_output_root, '{}_instantaneous_phase.npy'.format(file_basename))
    output_path_3 = os.path.join(npy_output_root, '{}_instantaneous_frequency.npy'.format(file_basename))
    np.save(output_path_1, amplitude)
    np.save(output_path_2, phase)
    np.save(output_path_3, frequency)
    pic_final_path = os.path.join(pic_output_root,'{}_Instantaneous_Charac'.format(file_basename))
    plot_feature_heatmaps(amplitude, phase, frequency,pic_final_path)
    
print("瞬时特征处理完成喵 !",end='\n')