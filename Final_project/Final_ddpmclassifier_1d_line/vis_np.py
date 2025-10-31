import numpy as np  
import cv2


read_path = "WavePrediction/train_data/2023-02-12-12-22-01-x_y_data-3.npy"
# 加载数据文件
data = np.load(read_path, allow_pickle=True)
seq_dict = data.item()

# 获取数据和元数据
seq_echoes = seq_dict.get('data')  # [64, 128, 128]
wave_height = seq_dict.get('height')
print(seq_echoes.shape,wave_height)