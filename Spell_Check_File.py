import numpy as np
import os

from Utils.Magic_Data_Check import test_read_npy, npy_value_check

if __name__ == "__main__":
    npy_file_path = "/media/ubuntu/Elements/Fuck_Data_ShitType5/Fucking_Instantaneous_npy/1_instantaneous_amplitude.npy"
    
    # ciallo: Npy file Dim Checking !
    test_read_npy(npy_file_path)
    
    # ciallo: Npy file Value Checking !
    # npy_value_check(npy_file_path)