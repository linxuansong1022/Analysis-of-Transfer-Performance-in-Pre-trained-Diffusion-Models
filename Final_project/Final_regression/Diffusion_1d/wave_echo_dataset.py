'''
制作波浪回波数据集。
'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os
from typing import Dict

from   scipy.fftpack    import fft, ifft, fft2, ifft2, fftn, ifftn
import matplotlib.pyplot as plt
from   torchvision      import transforms

#1. 每个*.npy文件中包含一个64x128x128的数据数组。
#2. 数据数组是3D数组，第一维是时间，第二维和第三维是空间。
#   从每个时间步随机选取16行，从每个npy文件随机选取8帧，因此每个npy文件得到8*16*128的数据。
#   将其转换为128*1*128的形式。这意味着批次大小是128，数据长度是128。

def measure_transform_errors(original: np.ndarray, transformed: np.ndarray) -> Dict[str, float]:
    """
    测量原始数据与其FFT-IFFT变换版本之间的误差

    Args:
        original: 原始数据数组
        transformed: IFFT变换后的数据数组

    Returns:
        包含不同误差指标的字典:
        - mse: 均方误差
        - rmse: 均方根误差
        - mae: 平均绝对误差
        - max_abs_error: 最大绝对误差
        - relative_error: 平均相对误差（由原始数据幅度归一化）
        - psnr: 峰值信噪比
    """
    # 确保数组具有相同的形状
    assert original.shape == transformed.shape, "Arrays must have the same shape"

    # 计算各种误差指标
    abs_diff = np.abs(original - transformed)
    squared_diff = (original - transformed) ** 2

    mse = np.mean(squared_diff)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    mae = np.mean(abs_diff)  # 平均绝对误差
    max_abs_error = np.max(abs_diff)  # 最大绝对误差

    # 计算相对误差，避免除以零
    mask = original != 0  # 创建非零元素的掩码
    relative_error = np.mean(abs_diff[mask] / np.abs(original[mask])) if np.any(mask) else float('inf')

    # 计算PSNR（峰值信噪比）
    max_pixel = max(np.max(original), np.max(transformed))
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse) if mse > 0 else float('inf')

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'max_abs_error': float(max_abs_error),
        'relative_error': float(relative_error),
        'psnr': float(psnr)
    }

def profile_hutan_data():
    '''
    增强时空图像并计算原始和增强后的频谱。
    保存增强后的文件供data_loader使用
    :return:
    '''
    # 训练文件路径（多个注释掉的路径供不同环境使用）
    # train_file_path = 'D:\Joey\Radar_Dataset\WaveMeasure_Shengshan_100w_shuffle_202111\\train_data\*.npy'
    # train_file_path = r'D:\Joey\Radar_Dataset\lvshun_dpca_20230227\normal\*.npy'
    # train_file_path = r'/home/ubuntu/datasets/wave_echos/lvshun_dpca_20230227/*.npy'
    train_file_path = r'D:\Dalian_LaohuTan_202302\WavePrediction_Dalian_LaohuTan_202302\train_data/*.npy'

    # 获取所有匹配的文件名
    train_file_names = glob.glob(train_file_path)

    # 创建PSNR可视化图形
    fig_psnr, ax_psnr = plt.subplots(figsize=(12, 6))
    fig_psnr.suptitle('PSNR Analysis Across Files and Frames')

    # 创建帧比较图形
    fig_frames, axs = plt.subplots(1, 2, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True, sharey=True)

    print('%d seq files in %s' % (len(train_file_names), train_file_path))
    wav_info_list = []
    # 初始化增强后的序列回波数组，维度顺序为了兼容tensorflow
    seq_echoes_aug = np.zeros((128,128,64)) #put the time_axis at last, for compatible with tensorflow

    # 初始化存储PSNR值的数组
    all_psnr_values = []
    file_indices = []
    frame_indices = []

    n = 0
    for file_name in train_file_names:
        # 获取不带路径的文件名
        title = os.path.basename(file_name)
        # 通过'-'分割文件名以获取日期时间组件
        parts = title.split('-')
        # 前6个部分包含日期和时间
        year = parts[0]
        month = parts[1]
        day = parts[2]
        hour = parts[3]
        minute = parts[4]
        second = parts[5]

        # 加载npy文件，允许pickle格式
        seq_dict = np.load(file_name, allow_pickle=True)
        # 打印文件索引和名称
        print(f'---{n}---: {file_name}')
        # 以yyyy-mm-dd hh:mm:ss格式打印日期
        print(f'{year}-{month}-{day} {hour}:{minute}:{second}')

        # 遍历字典中的所有键
        for key in seq_dict.item().keys():
            # 如果值是numpy数组，打印其形状
            if isinstance(seq_dict.item()[key], np.ndarray):
                print(key, seq_dict.item()[key].shape)
            # 否则打印键和值（保留.xx格式的精度）
            else:
                print(key, '{:.2f}'.format(seq_dict.item()[key]))

        # 从字典中获取各种数据
        seq_echoes_3dfft = seq_dict.item().get('fft')  # FFT变换后的数据
        seq_echoes_data  = seq_dict.item().get('data')  # 原始数据
        wave_height      = seq_dict.item().get('height')  # 波高
        wave_period      = seq_dict.item().get('period')  # 波周期
        wave_dir         = seq_dict.item().get('dir')  # 波向

        dpca_height      = seq_dict.item().get('dpca_height')  # DPCA高度
        assert(dpca_height == wave_height)  # 确保DPCA高度与波高一致
        seq_echoes       = np.real(ifftn(seq_echoes_3dfft))  # 执行逆FFT变换得到时域数据

        assert (np.sum(np.isnan(seq_echoes)) == 0)  # 确保数据中没有NaN值
        nframes = seq_echoes.shape[2]  # 获取帧数

        n += 1
        # 获取第一帧数据
        frame      = seq_echoes_data[:, :, 0]  # 原始数据的第一帧
        frame_ifft = seq_echoes[:, :, 0]  # IFFT变换后的第一帧

        # 显示原始数据和IFFT数据
        axs[0].imshow(frame),      axs[0].set_title('%s_rawdata'%(title))
        axs[1].imshow(frame_ifft), axs[1].set_title('frame_ifft')

        # 计算并打印每帧的误差指标
        print("\nError metrics between original and IFFT-transformed data:")
        for frame_idx in range(seq_echoes.shape[2]):
            # 计算原始数据和IFFT变换数据之间的误差
            errors = measure_transform_errors(
                seq_echoes_data[:,:,frame_idx],
                seq_echoes[:,:,frame_idx]
            )
            print(f"\nFrame {frame_idx}:")
            for metric, value in errors.items():
                print(f"{metric}: {value:.2e}")

            # 存储PSNR值用于可视化
            all_psnr_values.append(errors['psnr'])
            file_indices.append(n)
            frame_indices.append(frame_idx)

        # 显示第一帧比较
        frame = seq_echoes_data[:, :, 0]
        frame_ifft = seq_echoes[:, :, 0]
        axs[0].imshow(frame), axs[0].set_title(f'{title}\nrawdata')
        axs[1].imshow(frame_ifft), axs[1].set_title('frame_ifft')
        plt.draw()

        n += 1

    # 绘制PSNR值
    ax_psnr.scatter(file_indices, all_psnr_values, c=frame_indices, cmap='viridis', alpha=0.6)
    ax_psnr.set_xlabel('File Index')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.set_title('PSNR Distribution Across Files and Frames')

    # 添加帧索引的颜色条
    cbar = plt.colorbar(ax_psnr.collections[0], ax=ax_psnr)
    cbar.set_label('Frame Index')

    # 添加网格以提高可读性
    ax_psnr.grid(True, alpha=0.3)

    # 在30dB处添加水平线（良好质量的典型阈值）
    ax_psnr.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax_psnr.text(0, 30, '30 dB', color='r', va='bottom')

    plt.tight_layout()
    plt.show()


class WaveEchoDataset(Dataset):
    def __init__(self, data_dir, num_samples_per_file=8, num_rows_per_frame=16, transform=None):
        """
        波浪回波数据集类

        Args:
            data_dir (str): 包含.npy文件的目录
            num_samples_per_file (int): 每个文件随机采样的帧数，默认为8帧
            num_rows_per_frame (int): 每帧随机采样的行数，每帧随机采样的行数（默认16行）
            transform (callable, optional): 可选的数据变换函数
        """
        #保存初始化参数到实例变量中。
        self.data_dir = data_dir
        self.num_samples_per_file = num_samples_per_file  # 每个文件采样8帧
        self.num_rows_per_frame = num_rows_per_frame  # 每帧采样16行
        self.transform = transform

        # 获取目录中的所有.npy文件，如果没有找到任何文件，抛出异常。
        self.file_paths = glob.glob(data_dir)
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {data_dir}")

        # 计算总样本数：文件数量 × 每个文件的采样数。这不是单个文件中的总帧数，而是数据集会生成的总样本数。
        self.total_samples = len(self.file_paths) * num_samples_per_file
    #获取指定索引的样本，这是Dataset的核心方法。
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 计算文件索引，计算文件索引。例如：如果每个文件采样8个，那么idx=0-7对应第0个文件，idx=8-15对应第1个文件。
        file_idx = idx // self.num_samples_per_file

        # 加载数据文件
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        seq_dict = data.item()

        # 获取数据和元数据，从字典中获取数据和波高
        seq_echoes = seq_dict.get('data')  # [64, 128, 128] - 时间×空间×空间
        wave_height = seq_dict.get('height')  # 波高

        # 获取时间戳信息，从文件名中提取时间信息，并结合数据中的区域信息，创建一个人类可读的时间戳标识符，用于标记这个数据样本的来源和时间。
        title = os.path.basename(self.file_paths[file_idx])
        parts = title.split('-')
        (year, month, day, hour, minute, second) = parts[0:6]
        zone_id = seq_dict.get('zone_class')[0]  # 区域ID
        stamp = f'{year}-{month}-{day} {hour}:{minute}:{second} zone {zone_id}'

        # 数据标准化参数计算
        mean_data = np.mean(seq_echoes)#计算数组所有元素的平均值
        std_data = np.std(seq_echoes)#计算数组所有元素的标准差
        # 先选择帧，再选择行
        nframes = seq_echoes.shape[2]  # 获取总帧数，[128,128,64]，获取64帧
        # 随机选择8帧，不放回采样，确保不重复，选择范围[0, nframes-1]
        frame_indices = np.random.choice(nframes, self.num_samples_per_file, replace=False)
        # 对选中的帧，选中8帧，随机选择16行
        selected_frames = seq_echoes[:, :, frame_indices]  # [128, 128, 8]
        # 随机选择16行，seq_echoes.shape[0]：第一个维度的大小，行数128行
        # 从128行中随机选择16行
        row_indices = np.random.choice(seq_echoes.shape[0], self.num_rows_per_frame, replace=False)
        #使用行索引从selected_frames中提取数据
        #selected_frames[row_indices]：只选择特定的行
        #结果维度：[16, 128, 8]
        selected_data = selected_frames[row_indices]  # [16, 128, 8]

        # 转换为张量并调整维度
        selected_rows = torch.tensor(selected_data, dtype=torch.float32)#torch.tensor()：将NumPy数组转换为PyTorch张量，数据类型为32位浮点数
        selected_rows = selected_rows.permute(2, 0, 1)  # [8, 16, 128] - 交换维度顺序
        selected_rows = selected_rows.reshape(-1, 1, 128)  # [128, 1, 128] - 重塑为批次形式，第一维度为8*16=128，结果为[128, 1, 128]，即128个1×128的序列，1代表单通道

        # 标准化数据，标准正态分布（均值0，标准差1）
        selected_rows = (selected_rows - mean_data) / std_data

        # 复制波高信息，使其与数据批次大小匹配，将波高值转换为张量，重复128次
        wave_heights = torch.tensor(wave_height, dtype=torch.float32).repeat(selected_rows.shape[0])

        # 直接返回所有128个序列
        #返回三个值的元组：selected_rows：[128, 1, 128]的数据张量
        #wave_heights：128个波高值
        #stamp：时间戳字符串
        return selected_rows, wave_heights, stamp

    def get_metadata(self, idx):
        """
        获取特定样本的元数据

        Args:
            idx (int): 样本索引

        Returns:
            dict: 包含元数据的字典（高度、周期、方向）
        """
        file_idx = idx // self.num_samples_per_file
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        seq_dict = data.item()

        return {
            'height': seq_dict.get('height'),
            'period': seq_dict.get('period'),
            'direction': seq_dict.get('dir')
        }

# 测试数据集
if __name__ == '__main__':
    # 数据路径，只使用zone1的数据
    data_path = r'/tmp/pycharm_project_896/Dalian_LaohuTan_202302/WavePrediction_Dalian_LaohuTan_202302/train_data/*-1.npy'

    # 创建数据集实例
    dataset = WaveEchoDataset(data_dir=data_path)

    # 创建数据加载器，批次大小为10，打乱数据
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (data, height, stamps) in enumerate(dataloader):
        print("Batch:", i)
        print("Data shape:", data.shape)
        print("Height shape:", height.shape)
        print("Stamps (first few):", stamps[:min(5, len(stamps))])

        # 检查数据维度 [10, 128, 1, 128]，其中10是批次大小B
        Bsize = data.shape[0]  # 批次大小
        # 内部样本数量由数据集内部逻辑固定
        num_inner_samples = dataset.num_samples_per_file * dataset.num_rows_per_frame

        # 检查维度是否符合预期
        if data.shape[1] != num_inner_samples or height.shape[1] != num_inner_samples:
             print(f"Warning: Unexpected inner dimension sizes. Data: {data.shape[1]}, Height: {height.shape[1]}, Expected: {num_inner_samples}")

        # 重塑数据以匹配期望的格式
        data_reshaped = data.view(Bsize * data.shape[1], 1, -1)  # 展平为[B*128, 1, 128]
        height_reshaped = height.view(Bsize * height.shape[1], -1)
        if height_reshaped.shape[-1] != 1:
             height_reshaped = height_reshaped.unsqueeze(-1)  # 确保最后一维是1

        print("Reshaped data shape:", data_reshaped.shape)
        print("Reshaped height shape:", height_reshaped.shape)
        break  # 只测试第一个批次