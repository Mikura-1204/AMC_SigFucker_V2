#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Magic is used to verify a Fucking Paper Method.
#   Use Down-Dim Biospectrum Calculate Method to get AIB and SIB.
#   Finally splice to construct a Metric as Charac.
#
# - compute_third_order_cumulant: 计算三阶累积量矩阵
#   compute_bispectrum：基于三阶累积量矩阵计算双谱,谱的计算通过对累积量矩阵的每一项乘以相应的指数项然后求和实现
#   bispectrum 是大小为 (len(omega1_vals), len(omega2_vals)) 的复数矩阵，代表双谱的频域分布
#####------------------------------------------------------------------------------------#####

import numpy as np
from scipy.signal import csd
import os
from scipy.signal import welch, stft
from tqdm import tqdm

# ciallo: This Fuc is used to save IQ-Sig as Complex-Num
def My_Magic_IQ_Npysave_Complex(raw_npy_path, output_path):

    # ciallo GPT: 读取根目录下所有需要的文件
    raw_npy_list = os.listdir(raw_npy_path)
    for file_name in raw_npy_list:
        loaded_path = os.path.join(raw_npy_path, file_name)
        raw_npy_data = np.load(loaded_path)  # 替换为你的文件路径

        # ciallo: This form of IQ-Signal is just a plural splicing by TWO CHANNEL !
        #         [Pay Fucking Attention that its accuracy is still questionable !]
        iq_signal = raw_npy_data[:, 0, :] + 1j * raw_npy_data[:, 1, :]
        output_name = "IQ_File_{}".format(file_name)
        final_output_path = os.path.join(output_path, output_name)
        np.save(final_output_path, iq_signal)

# ciallo GPT: 假若是两通道IQ两路信号的话,可以将两路信号拼接喵
def compute_third_order_cumulant(signal, max_tau):
    """
    计算信号的三阶累积量 C3x(τ1, τ2)。
    
    参数：
    - signal: 输入信号，一维数组。
    - max_tau: 最大延迟，限制 τ1 和 τ2 的范围。
    
    返回：
    - cumulant_matrix: 三阶累积量矩阵，大小为 (2*max_tau+1, 2*max_tau+1)。
    """
    N = len(signal)
    cumulant_matrix = np.zeros((2 * max_tau + 1, 2 * max_tau + 1), dtype=np.complex128)
    
    for tau1 in range(-max_tau, max_tau + 1):
        for tau2 in range(-max_tau, max_tau + 1):
            sum_val = 0
            for t in range(max_tau, N - max_tau):
                try:
                    sum_val += signal[t] * np.conj(signal[t - tau1]) * np.conj(signal[t - tau2])
                except OverflowError:
                    print(f"溢出发生在 t={t}, tau1={tau1}, tau2={tau2}")
            cumulant_matrix[tau1 + max_tau, tau2 + max_tau] = sum_val / N  # 归一化处理
            
    return cumulant_matrix

def compute_bispectrum(signal, max_tau, omega1_vals, omega2_vals):
    """
    计算信号的双谱 Bx(ω1, ω2)。
    
    参数：
    - signal: 输入信号，一维数组。
    - max_tau: 最大延迟，控制 τ1 和 τ2 的范围。
    - omega1_vals: ω1 的取值数组。
    - omega2_vals: ω2 的取值数组。
    
    返回：
    - bispectrum: 双谱矩阵，大小为 (len(omega1_vals), len(omega2_vals))。
    """
    cumulant_matrix = compute_third_order_cumulant(signal, max_tau)
    bispectrum = np.zeros((len(omega1_vals), len(omega2_vals)), dtype=complex)
    
    for i, omega1 in enumerate(omega1_vals):
        for j, omega2 in enumerate(omega2_vals):
            sum_val = 0
            for tau1 in range(-max_tau, max_tau + 1):
                for tau2 in range(-max_tau, max_tau + 1):
                    exp_val = np.exp(-1j * (omega1 * tau1 + omega2 * tau2))
                    sum_val += cumulant_matrix[tau1 + max_tau, tau2 + max_tau] * exp_val
            bispectrum[i, j] = sum_val
            
    return bispectrum

def process_and_save_bispectrum(input_files, output_files, max_tau=10, omega_resolution=64):
    """
    读取每个输入文件，计算每个样本的I/Q双谱图并保存到输出文件。
    
    参数：
    - input_files: 输入文件路径列表。
    - output_files: 输出文件路径列表，长度需与input_files相同。
    - max_tau: 最大延迟。
    - omega_resolution: 双谱频率分辨率。
    """
    omega_vals = np.linspace(-np.pi, np.pi, omega_resolution)
    
    # ciallo GPT: 输入输出打包成迭代器
    for input_file, output_file in zip(input_files, output_files):
        data = np.load(input_file)  # 加载每个文件的数据，shape: [2000, 2, 2048]
        bispectra = []
        
        print(f"正在处理文件: {input_file} 喵")
        for i in tqdm(range(data.shape[0]), desc="样本进度"):  # 使用tqdm显示样本的进度
            iq_data = data[i]  # 获取第i个样本的IQ通道数据
            
            # 对I和Q通道分别计算双谱，并加入进度条显示
            bispectrum_I = compute_bispectrum(iq_data[0], max_tau, omega_vals, omega_vals)  # I通道
            bispectrum_Q = compute_bispectrum(iq_data[1], max_tau, omega_vals, omega_vals)  # Q通道
            
            # 将I和Q通道的双谱合并为一个样本的双谱结果
            bispectrum_sample = np.stack([bispectrum_I, bispectrum_Q], axis=0)  # shape: [2, omega_resolution, omega_resolution]
            bispectra.append(bispectrum_sample)
        
        # 保存每个文件的双谱结果为四维矩阵，shape: [2000, 2, omega_resolution, omega_resolution]
        bispectra = np.array(bispectra)
        np.save(output_file, bispectra)
        print(f"已保存双谱矩阵到文件: {output_file} 喵")

# ciallo: Get input and output file list     
def My_Magic_get_iofiles_list(input_path, output_path):
    input_namelist = []
    output_list = []
    input_list = []
    input_namelist = os.listdir(input_path)
    for input_file in input_namelist:
        file_basename = os.path.basename(input_file)
        filename_without_extension = os.path.splitext(file_basename)[0]
        output_name = os.path.join(output_path, "Biospectra_{}.npy".format(filename_without_extension))
        output_list.append(output_name)
        
        input_file_path = os.path.join(input_path,input_file)
        input_list.append(input_file_path)
    
    return input_list, output_list
