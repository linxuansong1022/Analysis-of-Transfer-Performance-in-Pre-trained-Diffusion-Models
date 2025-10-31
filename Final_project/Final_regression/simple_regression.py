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
try:
    from Diffusion_1d.wave_echo_dataset import WaveEchoDataset
except ImportError:
    print("错误：无法导入 WaveEchoDataset。请确保 Diffusion_1d 包在 Python 路径中。")
    exit()

# --- 回归模型 (支持单一线性层或 MLP 头) ---
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=128, head_type='linear', hidden_dims=[64, 32], dropout_rate=0.1):
        """
        初始化回归器。

        Args:
            input_dim (int): 输入序列的维度/长度。
            head_type (str): 回归头类型，可选 'linear' 或 'mlp'。
            hidden_dims (list): MLP头的隐藏层维度列表 (仅当 head_type='mlp' 时有效)。
            dropout_rate (float): MLP头中使用的Dropout率 (仅当 head_type='mlp' 时有效)。
        """
        super().__init__()
        self.head_type = head_type
        self.input_dim = input_dim

        if head_type == 'linear':
            # 单一线性层回归头
            self.regressor_head = nn.Linear(input_dim, 1)
            print(f"初始化 SimpleRegressor (Head: Linear, Input: {input_dim}, Output: 1)")
        elif head_type == 'mlp':
            # 三层 MLP 回归头
            layers = []
            in_d = input_dim
            # 添加隐藏层
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_d, h_dim))
                layers.append(nn.ReLU()) # 使用ReLU激活函数
                layers.append(nn.Dropout(dropout_rate)) # 添加Dropout
                in_d = h_dim
            # 添加最终输出层 (输出维度为1)
            layers.append(nn.Linear(in_d, 1))
            self.regressor_head = nn.Sequential(*layers)
            print(f"初始化 SimpleRegressor (Head: MLP, Input: {input_dim}, Hidden: {hidden_dims}, Output: 1, Dropout: {dropout_rate})")
        else:
            raise ValueError("无效的 head_type。请选择 'linear' 或 'mlp'。")

    def forward(self, x):
        # x 的输入形状预期是 [batch_size, input_dim]
        # 直接通过选定的回归头进行预测
        x = self.regressor_head(x)
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
            shuffle=False,
            num_workers=4,
            drop_last=True,
            pin_memory=True
        )
        print(f"验证集样本总数 (文件数 * samples_per_file): {len(val_dataset)}")
    except Exception as e:
        print(f"加载验证数据失败: {e}")
        return

    # --- 模型、损失函数、优化器 ---
    # 根据配置选择并初始化模型
    model = SimpleRegressor(
        input_dim=config["seq_length"],
        head_type=config["head_type"], # 从配置中获取头类型
        hidden_dims=config.get("mlp_hidden_dims", [64, 32]), # 可选配置 MLP 隐藏层
        dropout_rate=config.get("mlp_dropout", 0.1)         # 可选配置 MLP Dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    print(f"\n开始训练 (使用 {config['head_type']} 回归头)...")
    for epoch in range(config["epoch"]):
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        torch.cuda.empty_cache()
        gc.collect()

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epoch']} [Train]")
        for data, heights, _ in train_pbar:
            B, InnerB, _, L = data.shape
            inputs = data.view(B * InnerB, L).to(device)
            targets = heights.view(B * InnerB, 1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({"train_loss": f"{running_train_loss / train_batches:.5f}"}) # 调整格式

        avg_train_loss = running_train_loss / train_batches

        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        torch.cuda.empty_cache()
        gc.collect()

        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epoch']} [Val]")
        with torch.no_grad():
            for data, heights, _ in val_pbar:
                B, InnerB, _, L = data.shape
                inputs = data.view(B * InnerB, L).to(device)
                targets = heights.view(B * InnerB, 1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({"val_loss": f"{running_val_loss / val_batches:.5f}"}) # 调整格式

        avg_val_loss = running_val_loss / val_batches

        print(f"Epoch {epoch+1}/{config['epoch']}, Train MSE: {avg_train_loss:.5f}, Val MSE: {avg_val_loss:.5f}") # 调整格式

    print("训练完成!")

# --- 主程序入口 ---
if __name__ == '__main__':
    config = {
        "data_dir": "/root/autodl-tmp/WavePrediction_Dalian_LaohuTan_202302/train_data/*.npy",
        "val_data_dir": "/root/autodl-tmp/WavePrediction_Dalian_LaohuTan_202302/test_data/*.npy",
        "epoch": 200,
        "batch_size": 10,
        "lr": 1e-4,
        "seq_length": 128,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",

        # --- 新增配置：选择回归头类型 ---
        "head_type": "mlp",  # 在这里切换 'linear' 或 'mlp'

        # --- (可选) MLP 特定配置 ---
        # "mlp_hidden_dims": [64, 32], # 可以修改隐藏层维度
        # "mlp_dropout": 0.1           # 可以修改Dropout率
    }

    print(f"使用的设备: {config['device']}")
    run_simple_regression(config)
