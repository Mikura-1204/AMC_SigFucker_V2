import os
from Utils.Magic_PowerSpec import compute_power_spectrum,process_power_spectrum_heatmap

data_root_path = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_CutNpy2"
output_path_npy = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_Powerspec"
output_path_pic = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_Powerspec_Pic"
data_list = os.listdir(data_root_path)


for file in data_list:
    file_path = os.path.join(data_root_path,file)
    file_name, _ = os.path.splitext(file)
    npy_file = os.path.join(output_path_npy,file_name)
    pic_file = os.path.join(output_path_pic,file_name)
    powerspec = compute_power_spectrum(file_path)
    process_power_spectrum_heatmap(powerspec,npy_file,pic_file)

print("功率谱处理完成喵 !", end='\n')