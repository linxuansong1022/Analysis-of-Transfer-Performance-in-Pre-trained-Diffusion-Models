'''
make the dataset for wave echo.
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

#1. In each *.npy file, there is a 64x128x128 data array.
#2. The data array is a 3D array, the first dimension is the time, the second and third dimension are the space.
#   Take a random 16 rows in each time step and random 8 frames in each npy file, therefore for each npy file, we have 8*16*128 data. Transform it into the
#   form of 128*1*128. It means that the batch is 128 and the data length is 128.
def measure_transform_errors(original: np.ndarray, transformed: np.ndarray) -> Dict[str, float]:
    """
    Measure the errors between original data and its FFT-IFFT transformed version

    Args:
        original: Original data array
        transformed: IFFT transformed data array

    Returns:
        Dictionary containing different error metrics:
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - max_abs_error: Maximum Absolute Error
        - relative_error: Mean Relative Error (normalized by original data magnitude)
        - psnr: Peak Signal-to-Noise Ratio
    """
    # Ensure the arrays have the same shape
    assert original.shape == transformed.shape, "Arrays must have the same shape"

    # Calculate various error metrics
    abs_diff = np.abs(original - transformed)
    squared_diff = (original - transformed) ** 2

    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)

    # Calculate relative error, avoiding division by zero
    mask = original != 0
    relative_error = np.mean(abs_diff[mask] / np.abs(original[mask])) if np.any(mask) else float('inf')

    # Calculate PSNR
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
    Augmenate the time-spatial images and compute raw and augmentated spectrum.
    save augmentated files for data_loader
    :return:
    '''
    # train_file_path = 'D:\Joey\Radar_Dataset\WaveMeasure_Shengshan_100w_shuffle_202111\\train_data\*.npy'
    # train_file_path = r'D:\Joey\Radar_Dataset\lvshun_dpca_20230227\normal\*.npy'
    # train_file_path = r'/home/ubuntu/datasets/wave_echos/lvshun_dpca_20230227/*.npy'
    train_file_path = r'D:\Dalian_LaohuTan_202302\WavePrediction_Dalian_LaohuTan_202302\train_data/*.npy'

    train_file_names = glob.glob(train_file_path)

    # Create figure for PSNR visualization
    fig_psnr, ax_psnr = plt.subplots(figsize=(12, 6))
    fig_psnr.suptitle('PSNR Analysis Across Files and Frames')
    
    # Create figure for frame comparison
    fig_frames, axs = plt.subplots(1, 2, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True, sharey=True)
    
    print('%d seq files in %s' % (len(train_file_names), train_file_path))
    wav_info_list = []
    seq_echoes_aug = np.zeros((128,128,64)) #put the time_axis at last, for compatible with tensorflow

    # Initialize arrays to store PSNR values
    all_psnr_values = []
    file_indices = []
    frame_indices = []

    n = 0
    for file_name in train_file_names:
        # Get the filename without path
        title = os.path.basename(file_name)
        # Split the filename by '-' to get date-time components
        parts = title.split('-')
        # The first 6 parts contain the date and time
        year = parts[0]
        month = parts[1]
        day = parts[2]
        hour = parts[3]
        minute = parts[4]
        second = parts[5]

        # print(n, file_name)
        seq_dict = np.load(file_name, allow_pickle=True)
        # seq_dict = np.load(file_name)
        #print the file index and names 
        print(f'---{n}---: {file_name}')
        # print the data's date in yyyy-mm-dd hh:mm:ss format
        print(f'{year}-{month}-{day} {hour}:{minute}:{second}')
        for key in seq_dict.item().keys():
            #print the shape of the key, if it is a numpy array
            if isinstance(seq_dict.item()[key], np.ndarray):
                print(key, seq_dict.item()[key].shape)
            #else print the key and value
            else: #keep the precision of the value in .xx format
                print(key, '{:.2f}'.format(seq_dict.item()[key]))

        seq_echoes_3dfft = seq_dict.item().get('fft')
        seq_echoes_data  = seq_dict.item().get('data')
        wave_height      = seq_dict.item().get('height')
        wave_period      = seq_dict.item().get('period')
        wave_dir         = seq_dict.item().get('dir')

        dpca_height      = seq_dict.item().get('dpca_height')
        assert(dpca_height == wave_height)
        seq_echoes       = np.real(ifftn(seq_echoes_3dfft))
        #seq_echoes_aug = np.zeros_like(seq_echoes)
        assert (np.sum(np.isnan(seq_echoes)) == 0)  # make sure no isnan data in seq_echoes.
        nframes = seq_echoes.shape[2]
        # do frame normalization and augmentation in time_spatial domain
        # for i in range(nframes):
        #     frame      = seq_echoes_data[:, :, i]  # uti.array_normal()
        #     frame_ifft = seq_echoes[:, :, i]
        #     # frame = uti.array_normal(seq_echoes[:, :, i]) * 255  # some augmentation only available in uint8 array
        #     # frame = frame.astype(np.uint8)
        n += 1
        frame      = seq_echoes_data[:, :, 0]  # uti.array_normal()
        frame_ifft = seq_echoes[:, :, 0]
        axs[0].imshow(frame),      axs[0].set_title('%s_rawdata'%(title))
        axs[1].imshow(frame_ifft), axs[1].set_title('frame_ifft')
        # axs[1,0].imshow(kxky_raw),  axs[1,0].set_title('raw_kxky')
        # axs[1,1].imshow(kxky_aug),  axs[1,1].set_title('aug_kxky')


        # img_path = './kxky_spectrum_lvshun20230227/%s_aug_%02d.jpg' % (title[0:-4], t)
        # fig.savefig(img_path)

        # Calculate and print error metrics for each frame
        print("\nError metrics between original and IFFT-transformed data:")
        for frame_idx in range(seq_echoes.shape[2]):
            errors = measure_transform_errors(
                seq_echoes_data[:,:,frame_idx], 
                seq_echoes[:,:,frame_idx]
            )
            print(f"\nFrame {frame_idx}:")
            for metric, value in errors.items():
                print(f"{metric}: {value:.2e}")
            
            # Store PSNR values for visualization
            all_psnr_values.append(errors['psnr'])
            file_indices.append(n)
            frame_indices.append(frame_idx)
        
        # Display first frame comparison
        frame = seq_echoes_data[:, :, 0]
        frame_ifft = seq_echoes[:, :, 0]
        axs[0].imshow(frame), axs[0].set_title(f'{title}\nrawdata')
        axs[1].imshow(frame_ifft), axs[1].set_title('frame_ifft')
        plt.draw()
        
        n += 1

    # Plot PSNR values
    ax_psnr.scatter(file_indices, all_psnr_values, c=frame_indices, cmap='viridis', alpha=0.6)
    ax_psnr.set_xlabel('File Index')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.set_title('PSNR Distribution Across Files and Frames')
    
    # Add colorbar for frame indices
    cbar = plt.colorbar(ax_psnr.collections[0], ax=ax_psnr)
    cbar.set_label('Frame Index')
    
    # Add grid for better readability
    ax_psnr.grid(True, alpha=0.3)
    
    # Add horizontal line at 30dB (typical threshold for good quality)
    ax_psnr.axhline(y=30, color='r', linestyle='--', alpha=0.5)
    ax_psnr.text(0, 30, '30 dB', color='r', va='bottom')
    
    plt.tight_layout()
    plt.show()


class WaveEchoDataset(Dataset):
    def __init__(self, data_dir, num_samples_per_file=8, num_rows_per_frame=16, transform=None):
        """
        Dataset for wave echo data.
        
        Args:
            data_dir (str): Directory containing .npy files
            num_samples_per_file (int): Number of random frames to sample from each file
            num_rows_per_frame (int): Number of random rows to sample from each frame
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.num_samples_per_file = num_samples_per_file
        self.num_rows_per_frame = num_rows_per_frame
        self.transform = transform
        
        # Get all .npy files in the directory
        #self.file_paths = glob.glob(os.path.join(data_dir, '*-1.npy')) # using the data from zone1
        self.file_paths = glob.glob(data_dir)
        if not self.file_paths:
            raise ValueError(f"No .npy files found in {data_dir}")
            
        # Calculate total number of samples
        self.total_samples = len(self.file_paths) * num_samples_per_file
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 计算文件索引
        file_idx = idx // self.num_samples_per_file
        
        # 加载数据文件
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        seq_dict = data.item()
        
        # 获取数据和元数据
        seq_echoes = seq_dict.get('data')  # [64, 128, 128]
        wave_height = seq_dict.get('height')
        
        # 获取时间戳信息
        title = os.path.basename(self.file_paths[file_idx])
        parts = title.split('-')
        (year, month, day, hour, minute, second) = parts[0:6]
        zone_id = seq_dict.get('zone_class')[0]
        stamp = f'{year}-{month}-{day} {hour}:{minute}:{second} zone {zone_id}'
        
        # 数据标准化
        mean_data = np.mean(seq_echoes)
        std_data = np.std(seq_echoes)
        
        # 先选择帧，再选择行
        nframes = seq_echoes.shape[2]
        # 随机选择8帧
        frame_indices = np.random.choice(nframes, self.num_samples_per_file, replace=False)
        # 对选中的帧，随机选择16行
        selected_frames = seq_echoes[:, :, frame_indices]  # [64, 128, 8]
        
        # 随机选择16行
        row_indices = np.random.choice(seq_echoes.shape[0], self.num_rows_per_frame, replace=False)
        selected_data = selected_frames[row_indices]  # [16, 128, 8]
        
        # 转换为张量并调整维度
        selected_rows = torch.tensor(selected_data, dtype=torch.float32)
        selected_rows = selected_rows.permute(2, 0, 1)  # [8, 16, 128]
        selected_rows = selected_rows.reshape(-1, 1, 128)  # [128, 1, 128]
        
        # 标准化数据
        selected_rows = (selected_rows - mean_data) / std_data
        
        # 复制波高信息
        wave_heights = torch.tensor(wave_height, dtype=torch.float32).repeat(selected_rows.shape[0])
        
        # 直接返回所有128个序列
        return selected_rows, wave_heights, stamp
    
    def get_metadata(self, idx):
        """
        Get metadata for a specific sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing metadata (height, period, direction)
        """
        file_idx = idx // self.num_samples_per_file
        data = np.load(self.file_paths[file_idx], allow_pickle=True)
        seq_dict = data.item()
        
        return {
            'height': seq_dict.get('height'),
            'period': seq_dict.get('period'),
            'direction': seq_dict.get('dir')
        }
    
# test the dataset
if __name__ == '__main__':

    data_path = r'/tmp/pycharm_project_896/Dalian_LaohuTan_202302/WavePrediction_Dalian_LaohuTan_202302/train_data/*-1.npy' # using the data from zone1

    dataset = WaveEchoDataset(data_dir=data_path)
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset.get_metadata(0))
    #iterate the dataset with dataloader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (data, height, stamps) in enumerate(dataloader):

        print("Batch:", i)
        print("Data shape:", data.shape)
        print("Height shape:", height.shape)
        print("Stamps (first few):", stamps[:min(5, len(stamps))])

        # review the data as [128*B, 1, 128]
        Bsize = data.shape[0] # Bsize means the batch size
        # num_inner_samples �� dataset �ڲ��߼��̶���
        num_inner_samples = dataset.num_samples_per_file * dataset.num_rows_per_frame


        if data.shape[1] != num_inner_samples or height.shape[1] != num_inner_samples:
             print(f"Warning: Unexpected inner dimension sizes. Data: {data.shape[1]}, Height: {height.shape[1]}, Expected: {num_inner_samples}")


        data_reshaped = data.view(Bsize * data.shape[1], 1, -1)
        height_reshaped = height.view(Bsize * height.shape[1], -1)
        if height_reshaped.shape[-1] != 1:
             height_reshaped = height_reshaped.unsqueeze(-1)

        print("Reshaped data shape:", data_reshaped.shape)
        print("Reshaped height shape:", height_reshaped.shape)
        break