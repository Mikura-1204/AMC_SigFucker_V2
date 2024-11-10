import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_instantaneous_features(data):
    """
    提取信号的瞬时幅度、瞬时相位和瞬时频率特征，并处理异常值。
    参数:
    - data: np.ndarray，形状为 [样本数, 2, 样本长度]，包含IQ信号的二维数据
    
    返回:
    - instantaneous_amplitude, instantaneous_phase, instantaneous_frequency: 
      瞬时特征数组，分别表示幅度、相位、频率
    """
    samples, channels, length = data.shape

    # 初始化保存特征的数组
    instantaneous_amplitude = np.zeros((samples, length))
    instantaneous_phase = np.zeros((samples, length))
    instantaneous_frequency = np.zeros((samples, length - 1))

    for i in range(samples):
        i_channel = np.nan_to_num(data[i, 0, :])  # 使用 nan_to_num 处理 NaN 和 Inf 值
        q_channel = np.nan_to_num(data[i, 1, :])

        # 瞬时幅度
        amplitude = np.sqrt(i_channel**2 + q_channel**2)
        amplitude = np.nan_to_num(amplitude)  # 再次确保没有 NaN 值
        instantaneous_amplitude[i] = amplitude

        # 瞬时相位
        phase = np.arctan2(q_channel, i_channel)
        phase = np.nan_to_num(phase)  # 处理可能的 NaN 值
        instantaneous_phase[i] = phase

        # 瞬时频率 (相位差分)
        frequency = np.diff(phase)
        frequency = np.nan_to_num(frequency)
        instantaneous_frequency[i] = frequency

    return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

def plot_feature_heatmaps(instantaneous_amplitude, instantaneous_phase, instantaneous_frequency, save_path):
    """
    绘制每个瞬时特征的热力图，展示2000个样本的分布。
    参数:
    - instantaneous_amplitude, instantaneous_phase, instantaneous_frequency: np.ndarray
      分别表示幅度、相位、频率的特征矩阵
    """
    # 创建图形
    plt.figure(figsize=(18, 12))

    # 绘制瞬时幅度的热力图
    plt.subplot(3, 1, 1)
    sns.heatmap(instantaneous_amplitude, cmap="YlGnBu", cbar=True)
    plt.title("Instantaneous Amplitude Heatmap")
    plt.xlabel("Sample Length")
    plt.ylabel("Samples")

    # 绘制瞬时相位的热力图
    plt.subplot(3, 1, 2)
    sns.heatmap(instantaneous_phase, cmap="YlOrBr", cbar=True)
    plt.title("Instantaneous Phase Heatmap")
    plt.xlabel("Sample Length")
    plt.ylabel("Samples")

    # 绘制瞬时频率的热力图
    plt.subplot(3, 1, 3)
    sns.heatmap(instantaneous_frequency, cmap="coolwarm", cbar=True)
    plt.title("Instantaneous Frequency Heatmap")
    plt.xlabel("Sample Length - 1")
    plt.ylabel("Samples")

    plt.tight_layout()
    
    # 如果提供了 save_path，保存图片
    if save_path is not None:
        plt.savefig('{}.jpg'.format(save_path), format='jpg')