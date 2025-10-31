# simple_regression.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import gc

# 假设 WaveEchoDataset 在 Diffusion_1d 包中
from Diffusion_1d.wave_echo_dataset import WaveEchoDataset

# --- 简单的回归模型 (修改后，直接使用类似 diffusion_1d 中的回归头结构) ---
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=128): # 输入维度是序列长度
        super().__init__()
        # 直接使用一个线性层作为回归头，输入维度为序列长度，输出为1
        # 这模拟了 diffusion_1d.py 中 self.fc = nn.Linear(128, 1) 的结构
        # 但请注意，这里的输入是原始序列，而不是提取的特征
        self.fc_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x 的输入形状预期是 [batch_size, sequence_length=128]
        # 直接通过回归头进行预测
        x = self.fc_head(x)
        return x

# --- 训练函数 ---
def run_simple_regression(config):
    device = torch.device(config["device"])

    # --- 数据加载 ---
    print("加载训练数据集...")
    try:
        train_dataset = WaveEchoDataset(
            data_dir=config["data_dir"],
            num_samples_per_file=8,
            num_rows_per_frame=16
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        print(f"训练集样本总数 (文件数 * samples_per_file): {len(train_dataset)}")
        # 检查第一个批次的数据形状
        # first_batch_data, first_batch_heights, _ = next(iter(train_dataloader))
        # print(f"原始训练批次数据形状: {first_batch_data.shape}") # 预期: [batch_size, 128, 1, 128]
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        return

    print("加载验证数据集...")
    try:
        val_dataset = WaveEchoDataset(
            data_dir=config["val_data_dir"],
            num_samples_per_file=8,
            num_rows_per_frame=16
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False, # 验证集通常不打乱
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        print(f"验证集样本总数 (文件数 * samples_per_file): {len(val_dataset)}")
    except Exception as e:
        print(f"加载验证数据失败: {e}")
        return

    # --- 模型、损失函数、优化器 ---
    model = SimpleRegressor(input_dim=config["seq_length"]).to(device)
    criterion = nn.MSELoss() # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    print("\n开始训练...")
    for epoch in range(config["epoch"]):
        model.train() # 设置为训练模式
        running_train_loss = 0.0
        train_batches = 0

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epoch']} [Train]")
        for data, heights, _ in train_pbar:
            # data 形状: [batch_size, 128, 1, 128] (B, InnerB, C, L)
            # heights 形状: [batch_size, 128]
            B, InnerB, _, L = data.shape

            # 重塑数据以匹配模型输入: [B * InnerB, L]
            inputs = data.view(B * InnerB, L).to(device)
            # 重塑目标: [B * InnerB, 1]
            targets = heights.view(B * InnerB, 1).to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({"train_loss": running_train_loss / train_batches})

        avg_train_loss = running_train_loss / train_batches

        # --- 验证阶段 ---
        model.eval() # 设置为评估模式
        running_val_loss = 0.0
        val_batches = 0
        torch.cuda.empty_cache()
        gc.collect()

        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epoch']} [Val]")
        with torch.no_grad(): # 在验证阶段不计算梯度
            for data, heights, _ in val_pbar:
                B, InnerB, _, L = data.shape
                inputs = data.view(B * InnerB, L).to(device)
                targets = heights.view(B * InnerB, 1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({"val_loss": running_val_loss / val_batches})

        avg_val_loss = running_val_loss / val_batches

        print(f"Epoch {epoch+1}/{config['epoch']}, Train MSE: {avg_train_loss:.4f}, Val MSE: {avg_val_loss:.4f}")

    print("训练完成!")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 使用与 Main_1d_reg.py 相似的配置
    config = {
        # 数据路径 (请确保这些路径在您的环境中是正确的)
        "data_dir": "/root/autodl-tmp/WavePrediction/train_data/1/*.npy",
        "val_data_dir": "/root/autodl-tmp/WavePrediction/test_data/*.npy",

        # 训练参数 (与 Main_1d_reg.py 对齐)
        "epoch": 200,       # 更新为 200
        "batch_size": 10,   # 更新为 10
        "lr": 1e-4,         # 更新为 1e-4
        "seq_length": 128,  # 与 WaveEchoDataset 输出一致
        "device": "cuda:0" if torch.cuda.is_available() else "cpu", # 保持动态选择，优先 cuda:0
    }

    print(f"使用的设备: {config['device']}")
    run_simple_regression(config)
