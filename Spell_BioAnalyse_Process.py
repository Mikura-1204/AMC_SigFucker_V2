import numpy as np
import os
from Utils.Magic_BioSpec import My_Magic_get_iofiles_list, process_and_save_bispectrum
from Utils.Magic_Value_Check import get_nonzero_values


if __name__ == "__main__":
    IQ_npy_path = "/media/star/Elements/Fuck_Data_ShitType5/Fucking_CutNpy2"
    biospec_npy_path = "/media/star/Elements/Fuck_Data_ShitType5/Fucking_Biospec"
      
    # ciallo: Prepare file list
    input_files, output_files = My_Magic_get_iofiles_list(IQ_npy_path, biospec_npy_path)
    print("输入文件列表如下喵: ",input_files,end='\n')
    print("输出文件列表如下喵: ",output_files,end='\n')
    
    # ciallo: Bio-Analyse Process
    process_and_save_bispectrum(input_files, output_files)
    print("双谱特征计算完成喵 !", end='\n')