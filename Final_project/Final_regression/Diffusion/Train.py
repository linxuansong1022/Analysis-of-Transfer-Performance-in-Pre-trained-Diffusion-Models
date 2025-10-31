import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionClassifyTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.ToTensor(),  # 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量 (将像素值缩放到 [0, 1])
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 将张量归一化到 [-1, 1] 范围，使用均值和标准差 (0.5, 0.5, 0.5)
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # 创建DataLoader，用于批量加载数据
    # shuffle=True: 每个epoch开始时打乱数据顺序
    # num_workers: 使用多个子进程加载数据，加快速度
    # drop_last=True: 如果最后一个批次数据量不足batch_size，则丢弃
    # pin_memory=True: 将数据加载到内存的固定区域，可以加速CPU到GPU的数据传输

    # 实例化 U-Net 模型，传入配置参数 T (时间步数), ch (基础通道数) 等
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:  # 检查是否需要加载预训练权重
        net_model.load_state_dict(torch.load(os.path.join(  # 加载指定的权重文件
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))

    # --- 优化器和学习率调度器设置 ---
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    # 创建AdamW优化器，用于更新UNet模型的所有参数
    # lr: 学习率
    # weight_decay: 权重衰减（L2正则化），防止过拟合
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    # 创建余弦退火学习率调度器，用于在训练过程中动态调整学习率
    # T_max: 最大训练轮数
    # eta_min: 最小学习率
    # last_epoch: 上一个epoch的索引，-1表示从头开始
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    # 创建预热学习率调度器，用于在训练开始时逐渐增加学习率
    # optimizer: 优化器
    # multiplier: 学习率预热倍数
    # warm_epoch: 预热轮数
    # after_scheduler: 预热结束后使用的调度器

    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    # 创建 GaussianDiffusionTrainer 对象，用于训练 DDPM 模型
    # net_model: 神经网络模型
    # beta_1: 初始噪声水平
    # beta_T: 最终噪声水平
    # T: 时间步数

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # labels 在这个标准 DDPM 训练中未使用，但 DataLoader 会返回它，因为这里是做特征提取，无监督
                # 1. 梯度清零，在计算新的梯度之前，必须清除上一个批次计算得到的旧梯度，这是 PyTorch 训练循环的标准步骤。
                optimizer.zero_grad()
                # 2. 将输入图像移动到设备
                x_0 = images.to(device)
                # 3. 通过训练器进行前向传播，计算损失
                # trainer(x_0) 内部会：随机采样时间步 t -> 计算带噪图像 x_t -> U-Net 预测噪声 -> 计算 MSE 损失
                loss = trainer(x_0).sum() / 1000.
                # 4. 反向传播计算梯度
                loss.backward()  # 计算出的梯度不会作为返回值，而是会积累在每个参数的 .grad 属性中，所以每次迭代需要清零
                # 5. (可选) 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                # 6. 更新模型参数，指示优化器根据已经计算好的梯度来更新模型的参数，化器会使用其内部算法（如 AdamW）和当前的学习率来执行更新
                optimizer.step()
                # 7. 更新 tqdm 进度条的后缀信息，显示当前状态
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()  # 在每个 epoch 完成后，调用学习率调度器 `warmUpScheduler` 的 `step` 方法，预热 + 余弦退火
        torch.save(net_model.state_dict(), os.path.join(  # 保存当前 epoch 的模型权重
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


# --- 分类器训练函数 (目标：在冻结的DDPM特征提取器上训练分类头) ---
def train_classify(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # 数据集
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    train_dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    dataset = CIFAR10(
        root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    val_dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    print("train batches:", len(train_dataloader), "val batches:", len(val_dataloader))

    # 模型设置：构建 DDPM 模型，并加载预训练权重
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    # 创建 UNet 模型，用于特征提取
    # T: 时间步数
    # ch: 特征通道数
    # ch_mult: 特征通道数倍率
    # attn: 是否使用注意力机制
    # num_res_blocks: 残差块数量
    # dropout: 随机失活率

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(modelConfig["training_load_weight"], map_location=device))
        print("Load pretrain")

    # 创建 GaussianDiffusionClassifyTrainer 对象，用于训练分类头
    # net_model: 特征提取器
    # beta_1: 初始噪声水平
    # beta_T: 最终噪声水平
    # T: 时间步数
    # head: 分类头
    # classify_T: 分类头时间步数
    trainer = GaussianDiffusionClassifyTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        modelConfig["head"], modelConfig["classify_T"]
    ).to(device)

    # 创建 AdamW 优化器，用于更新分类头
    # trainer.fc.parameters(): 分类头参数
    # lr: 学习率
    # weight_decay: 权重衰减（L2正则化），防止过拟合
    optimizer = torch.optim.AdamW(
        trainer.fc.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)

    # 创建余弦退火学习率调度器，用于在训练过程中动态调整学习率
    # T_max: 最大训练轮数
    # eta_min: 最小学习率
    # last_epoch: 上一个epoch的索引，-1表示从头开始
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)

    # 创建预热学习率调度器，用于在训练开始时逐渐增加学习率
    # optimizer: 优化器
    # multiplier: 学习率预热倍数
    # warm_epoch: 预热轮数
    # after_scheduler: 预热结束后使用的调度器
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    # 冻结 DDPM 模型参数
    print("Freeze DDPM model parameters...")
    for child in list(net_model.children()):  # 遍历 UNet 模型的所有子模块
        for param in child.parameters():  # 遍历每个子模块的参数
            param.requires_grad = False  # 设置参数的 requires_grad 属性为 False，表示这些参数不需要参与梯度更新

    # 打印模型参数数量
    def print_params(model, name):
        total_params = sum(p.numel() for p in model.parameters())  # 计算模型参数的总数量
        print(f"{name} 参数量: {total_params}")  # 打印模型参数数量

    print_params(net_model, "预训练DDPM模型")
    print_params(trainer.fc, "分类器")

    # 开始训练
    for e in range(modelConfig["epoch"]):
        # --- 训练阶段 ---
        net_model.train()
        trainer.fc.train()
        train_correct = 0
        # 使用 tqdm 包装训练数据加载器，并添加描述
        with tqdm(train_dataloader, dynamic_ncols=True,
                  desc=f"Epoch {e} [Train]") as tqdmDataLoader:  # tqdmDataLoader 是训练数据加载器
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()  # 梯度清零，在计算新的梯度之前，必须清除上一个批次计算得到的旧梯度
                x_0 = images.to(device)  # 将输入图像移动到设备
                labels = labels.to(device)  # 将标签移动到设备
                pre, loss = trainer(x_0, labels)  # 前向传播，计算损失
                loss = loss.sum() / 1000.  # 缩放损失
                loss.backward()  # 反向传播，计算梯度

                pred = nn.Softmax(dim=1)(pre)  # 使用 Softmax 函数将预测结果转换为概率分布
                pred = pred.max(1, keepdim=True)[1]  # 获取最大概率的类别
                train_correct += pred.eq(labels.view_as(pred)).sum().item()  # 计算正确预测的数量

                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])  # 梯度裁剪，防止梯度爆炸
                optimizer.step()  # 更新模型参数
                # 设置训练进度条后缀信息
                # ordered_dict: 有序字典，用于存储训练过程中的各种信息
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        # --- Epoch 训练结束后 ---
        warmUpScheduler.step()  # 在每个 epoch 完成后，调用学习率调度器 `warmUpScheduler` 的 `step` 方法，预热 + 余弦退火
        torch.save(net_model.state_dict(),  # 保存当前 epoch 的模型权重
                   os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

        train_acc = train_correct / len(train_dataloader.dataset)  # 计算训练准确率

        # --- 评估阶段 ---
        net_model.eval()  # 设置模型为评估模式，不进行梯度更新
        trainer.fc.eval()  # 设置分类头为评估模式，不进行梯度更新
        val_correct = 0  # 初始化验证正确预测的数量
        val_loss_sum = 0  # 初始化验证损失的累加值
        with torch.no_grad():  # 禁用梯度计算，在验证阶段不需要计算梯度
            # ***** 修改开始：使用 tqdm 包装验证加载器 *****
            with tqdm(val_dataloader, dynamic_ncols=True, desc=f"Epoch {e} [Val]") as tqdmValLoader:
                for images, labels in tqdmValLoader:  # 遍历带进度条的验证加载器
                    x_0 = images.to(device)  # 将输入图像移动到设备
                    labels = labels.to(device)  # 将标签移动到设备
                    pre, loss = trainer(x_0, labels)  # 前向传播，计算损失
                    batch_loss = loss.sum().item()  # 获取当前批次的损失值
                    val_loss_sum += batch_loss  # 累加损失

                    pred = nn.Softmax(dim=1)(pre)  # 使用 Softmax 函数将预测结果转换为概率分布
                    pred = pred.max(1, keepdim=True)[1]  # 获取最大概率的类别
                    val_correct += pred.eq(labels.view_as(pred)).sum().item()  # 计算正确预测的数量

                    tqdmValLoader.set_postfix(ordered_dict={
                        "epoch": e,  # 当前 Epoch
                        "loss: ": batch_loss / 1000.,  # 当前批次的损失 (与训练时一样进行缩放显示)
                        "img shape: ": x_0.shape,  # 输入图像形状
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]  # 当前优化器的学习率 (虽然验证时不更新，但保持格式一致)
                    })

        val_acc = val_correct / len(val_dataloader.dataset)

        # 修改打印语句以包含平均验证损失
        print(f"Epoch: {e}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
