import numpy as np
import matplotlib.pyplot as plt
import os

def compute_power_spectrum(file_path):
    # 计算数据的功率谱
    # 假设 data 的维度是 (samples, channels, length)，这里我们对每个样本计算功率谱
    # 载入数据
    data = np.load(file_path)
    
    # 检查数据是否成功加载
    if data is None or data.ndim != 3:
        print(f"数据加载失败或数据维度错误喵，文件路径：{file_path}，加载结果维度：{None if data is None else data.shape}")
        return None
    
    # 计算数据的功率谱
    # 对 data 的最后一个维度（即时间序列长度维度）进行快速傅里叶变换。np.fft.fft 返回的是复数数组，表示信号在频域上的表示。
    # 对 FFT 结果取模（即计算复数的绝对值），然后平方。这是计算功率谱的标准方法。功率谱表示信号在不同频率上的能量分布。
    power_spectrum = np.abs(np.fft.fft(data, axis=-1))**2
    return power_spectrum

def process_power_spectrum_heatmap(power_spectrum, output_npy_path, output_jpg_path):
    # 计算所有样本的功率谱的平均值
    # 假设 power_spectrum 维度是 (samples, channels, length)，
    # 这里我们对所有样本2000个样本取平均得到热力图
    # 形状为 (channels, length // 2 + 1)。每个元素表示所有样本在一个通道的一个频率点上的平均功率谱值
    avg_power_spectrum = np.mean(power_spectrum, axis=0)  # 按样本求平均，得到 (channels, length)

    
    # 对数缩放功率谱,让数值变大一些容易观察
    avg_power_spectrum_log = np.log10(avg_power_spectrum)
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    
    # ciallo GPT: 设置图像的坐标范围。extent 参数是一个四元组 [left, right, bottom, top]，分别对应图像在 x 轴和 y 轴上的范围。
    #             在这个例子中，x 轴的范围是从 0 到 avg_power_spectrum.shape[1]，y 轴的范围是从 0 到 avg_power_spectrum.shape[0]。
    #             这意味着图像的宽度和高度将分别对应于 avg_power_spectrum 的列数和行数
    plt.imshow(avg_power_spectrum_log, aspect='auto', cmap='hot', origin='lower', 
               extent=[0, avg_power_spectrum.shape[1], 0, avg_power_spectrum.shape[0]],
               vmin=-2, vmax=np.max(avg_power_spectrum_log))
    plt.colorbar(label="Power")
    plt.title("Power Spectrum Heatmap (IQ Channel 1)")
    plt.xlabel("Frequency bins")
    plt.ylabel("Channels (IQ)")
    plt.tight_layout()

    # 保存热力图为 jpg 文件
    plt.savefig(output_jpg_path)
    plt.close()

    # 保存平均功率谱为 npy 文件
    np.save(output_npy_path, avg_power_spectrum)