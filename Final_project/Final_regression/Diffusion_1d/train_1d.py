import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
# import wandb  # <-- 1. Import wandb
import gc

# 设置CUDA内存分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scheduler import GradualWarmupScheduler

# 从同一个包内导入模块
from .wave_echo_dataset import WaveEchoDataset
from .model_1d import UNet1D
from .diffusion_1d import GaussianDiffusionTrainer1D, GaussianDiffusionSampler1D,GaussianDiffusionTrainer1DReg

logger = logging.getLogger(__name__)


class Synthetic1DDataset(Dataset):
    def __init__(self, num_samples=10000, seq_length=128):
        self.num_samples = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic 1D data (you can modify this to your specific data)
        # Example: Generate a sine wave with random frequency and phase
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        x = np.linspace(0, 10, self.seq_length)
        y = np.sin(freq * x + phase)

        # Add some noise
        y += np.random.normal(0, 0.1, self.seq_length)

        # Normalize to [-1, 1]
        y = (y - y.min()) / (y.max() - y.min()) * 2 - 1

        return torch.FloatTensor(y).unsqueeze(0)  # Add channel dimension


def train(modelConfig: dict):
    device = torch.device(modelConfig["device"])
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置梯度累积步数
    accumulation_steps = 4
    
    # 添加训练损失监控
    train_losses = []
    best_loss = float('inf')
    patience = 20  # 早停的耐心值
    patience_counter = 0
    
    # try:
    #     # 初始化wandb，添加更多配置项
    #     run = wandb.init(
    #         project="ddpm-wave-1d",
    #         config={
    #             **modelConfig,
    #             "architecture": "UNet1D",
    #             "dataset": "WaveEcho",
    #             "optimizer": "AdamW",
    #             "learning_rate": modelConfig["lr"],
    #             "scheduler": "WarmupCosine"
    #         },
    #         name=f"ddpm_wave_1d_{wandb.util.generate_id()}",
    #         resume="allow"
    #     )
        
    #     # 创建wandb自定义图表
    #     wandb.define_metric("epoch")
    #     wandb.define_metric("train_loss", step_metric="epoch")
    #     wandb.define_metric("learning_rate", step_metric="epoch")
    #     wandb.define_metric("batch_loss", step_metric="global_step")
        
    #     logger.info("Weights & Biases initialized successfully.")
    # except Exception as e:
    #     logger.error(f"Failed to initialize Weights & Biases: {e}")
    #     run = None

    # 加载数据集
    dataset = WaveEchoDataset(
        data_dir=modelConfig["data_dir"],
        num_samples_per_file=8,
        num_rows_per_frame=16
    )
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],  # 使用 modelConfig 中的配置
        shuffle=True, #打乱数据
        num_workers=4,#使用工作进程
        drop_last=True,
        pin_memory=True
    )

    # 初始化模型
    net_model = UNet1D(
        T=modelConfig["T"],#时间步
        ch=modelConfig["channel"],#初始通道数
        ch_mult=modelConfig["channel_mult"],#通道倍增
        attn=modelConfig["attn"],#attn是选择哪一层使用自注意力机制
        num_res_blocks=modelConfig["num_res_blocks"],#每一层的残差块数量
        dropout=modelConfig["dropout"]#丢弃率
    ).to(device)

    # 加载预训练权重（如果指定）
    if modelConfig["train_load_weight"] is not None:
        weight_path = os.path.join(modelConfig["save_weight_dir"], modelConfig["train_load_weight"])
        try:
            net_model.load_state_dict(torch.load(weight_path, map_location=device))
            logger.info(f"成功加载预训练权重：{weight_path}")
            start_epoch = int(modelConfig["train_load_weight"].split('_')[1])
            logger.info(f"将从epoch {start_epoch} 继续训练")
        except Exception as e:
            logger.error(f"加载预训练权重失败：{e}")
            logger.info("将从头开始训练...")
            start_epoch = 0
    else:
        start_epoch = 0
        logger.info("从头开始训练...")

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        net_model.parameters(),#模型传入的参数
        lr=modelConfig["lr"],#学习率
        weight_decay=1e-4 #权重衰减
    )

    # 设置学习率调度器
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,#传入优化器
        T_max=modelConfig["epoch"],#余弦退火的周期
        eta_min=0,#最小学习率
        last_epoch=-1#上一个epoch的索引
    )
    
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],#学习率的倍增因子
        warm_epoch=modelConfig["epoch"] // 10,#热身的epoch数
        after_scheduler=cosineScheduler#热身结束后接一个余弦退火调度器
    )

    # 初始化训练器
    trainer = GaussianDiffusionTrainer1D(
        net_model,
        modelConfig["beta_1"],#起始噪声系数
        modelConfig["beta_T"],
        modelConfig["T"]
    ).to(device)

    # 创建保存目录
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    
    # 创建训练损失图表保存目录
    loss_plot_dir = os.path.join(modelConfig["save_weight_dir"], "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)

    # 开始训练
    global_step = 0
    for e in range(start_epoch, modelConfig["epoch"]):
        epoch_loss = 0.0#初始化每个 epoch 的总损失
        num_batches = 0#初始化每个 epoch 的批次数量
        
        # 每个epoch开始时清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            tqdmDataLoader.set_description(f"Epoch {e+1}/{modelConfig['epoch']}")
            
            for batch_idx, (data, heights, stamps) in enumerate(tqdmDataLoader):
                # 重塑数据以适应模型
                B = data.shape[0]
                x_0 = data.view(B*128, 1, 128).to(device)# 将数据调整为 (B*128, 1, 128) 的形状，并移动到指定设备上
                heights = heights.view(B*128).to(device) #无监督学习，并没有使用到标签heights
                
                # 计算损失并进行梯度累积
                current_loss = trainer(x_0).mean()# 调用训练器计算当前批次的损失
                loss = current_loss * 1000. / accumulation_steps   #除以累积步数，得到调整后的损失

                # 反向传播
                loss.backward()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(#梯度裁剪，防止梯度爆炸
                        net_model.parameters(),
                        modelConfig["grad_clip"]
                    )
                    optimizer.step() # 更新模型参数
                    optimizer.zero_grad()# 清空梯度
                
                # 更新进度条
                current_lr = optimizer.state_dict()['param_groups'][0]["lr"]# 获取当前的学习率
                epoch_loss += current_loss.item()# 累加当前批次的损失到 epoch 总损失中
                num_batches += 1# 批次数量加 1
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                # 更新进度条的后缀信息
                tqdmDataLoader.set_postfix(ordered_dict={
                    "batch_loss": f"{current_loss.item():.4f}",
                    "LR": f"{current_lr:.6f}"
                })

                # # 记录到wandb - 批次级别的指标
                # if run:
                #     try:
                #         wandb.log({
                #             "batch_loss": current_loss.item(),
                #             "learning_rate": current_lr,
                #             "global_step": global_step,
                #         })
                #     except Exception as log_e:
                #         logger.warning(f"Failed to log batch metrics to Weights & Biases: {log_e}")

                global_step += 1

        # 每个epoch结束时清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 计算并记录每个epoch的平均损失
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_epoch_loss)# 将平均损失添加到 train_losses 列表中
        
        # 记录到wandb - epoch级别的指标
        # if run:
        #     try:
        #         wandb.log({
        #             "epoch": e,
        #             "train_loss": avg_epoch_loss,
        #             "learning_rate": current_lr,
        #             "best_loss": best_loss,
        #             "patience_counter": patience_counter
        #         })
        #     except Exception as log_e:
        #         logger.warning(f"Failed to log epoch metrics to Weights & Biases: {log_e}")

        logger.info(f"Epoch {e+1} finished. Average Loss: {avg_epoch_loss:.4f}")

        # 早停检查
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(modelConfig["save_weight_dir"], 'best_model.pt')
            torch.save(net_model.state_dict(), best_model_path)
            logger.info(f"保存最佳模型，损失值：{best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！连续{patience}个epoch没有改善。")
                break

        # 绘制训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存损失曲线图
        loss_plot_path = os.path.join(loss_plot_dir, f'loss_plot_epoch_{e+1}.png')
        plt.savefig(loss_plot_path)
        plt.close()

        

        # 保存检查点
        save_path = os.path.join(modelConfig["save_weight_dir"], f'ckpt_{e}_.pt')
        try:
            torch.save(net_model.state_dict(), save_path)
            logger.info(f"Epoch {e}: Checkpoint saved to {save_path}")
        except Exception as save_error:
            logger.error(f"Epoch {e}: Failed to save checkpoint: {save_error}")

        # 定期生成样本
        if e % modelConfig.get("sample_interval", 10) == 0:
            net_model.eval()
            sampler = GaussianDiffusionSampler1D(
                net_model,
                modelConfig["beta_1"],
                modelConfig["beta_T"],
                modelConfig["T"]
            ).to(device)

            with torch.no_grad():
                x_T = torch.randn(
                    [modelConfig["nrow"], 1, 128],  # 直接使用正确的3D形状
                    device=device
                )
                x_0 = sampler(x_T)

                # 绘制样本
                fig = plt.figure(figsize=(10, 10))
                fig.suptitle(f"Epoch {e} Samples")
                for i in range(modelConfig["nrow"]):
                    ax = fig.add_subplot(modelConfig["nrow"] // 2, 2, i + 1)
                    ax.plot(x_0[i, 0].cpu().numpy())  # 直接使用正确的维度
                    ax.set_ylim(-1, 1)
                    ax.set_title(f"Sample {i+1}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # 保存采样图像
                sample_path = os.path.join(modelConfig["sampled_dir"], f"sample_{e}.png")
                try:
                    plt.savefig(sample_path)
                    logger.info(f"Epoch {e}: Sample image saved to {sample_path}")
                except Exception as img_save_e:
                    logger.error(f"Epoch {e}: Failed to save sample image: {img_save_e}")

                # # 记录到wandb
                # if run:
                #     try:
                #         wandb.log({
                #             "samples": wandb.Image(plt),
                #             "train_loss": avg_epoch_loss,
                #             "best_loss": best_loss
                #         }, step=global_step)
                #     except Exception as log_img_e:
                #         logger.warning(f"Failed to log sample image to Weights & Biases: {log_img_e}")

                plt.close(fig)

            net_model.train()
        # 更新学习率
        warmUpScheduler.step()

    # 保存最终的训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Final Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    final_loss_plot_path = os.path.join(loss_plot_dir, 'final_loss_plot.png')
    plt.savefig(final_loss_plot_path)
    plt.close()

    # # 结束wandb运行
    # if run:
    #     try:
    #         wandb.finish()
    #         logger.info("Weights & Biases run finished.")
    #     except Exception as finish_e:
    #         logger.error(f"Error finishing Weights & Biases run: {finish_e}")


def train_reg(modelConfig: dict):
    device = torch.device(modelConfig["device"])
    
    # 清理GPU缓存，释放不必要的内存，避免内存溢出问题
    torch.cuda.empty_cache()
    gc.collect()# 强制进行垃圾回收，释放不再使用的对象占用的内存
    
    # 设置梯度累积的步数。梯度累积允许在多个小批次上累积梯度，然后再进行一次参数更新，相当于增大了有效批次大
    accumulation_steps = 4
    
    # 添加训练损失监控
    train_losses = [] # 初始化一个空列表，用于记录每个epoch的训练损失，方便后续分析和可视化
    best_loss = float('inf') # 初始化最佳损失为正无穷大，用于后续比较验证集损失，找出最小损失对应的模型
    patience = 20  # 设置早停的耐心值，即允许验证集损失连续多少个epoch没有改善才停止训练
    patience_counter = 0# 初始化早停计数器，用于记录验证集损失连续没有改善的epoch数量
    


    # 加载训练数据集
    dataset = WaveEchoDataset(
        data_dir=modelConfig["data_dir"],
        num_samples_per_file=8,
        num_rows_per_frame=16
    )
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],  # 使用 modelConfig 中的配置
        shuffle=True,#打乱
        num_workers=4,#4个线程
        drop_last=True,#丢弃最后未满的批次
        pin_memory=True#提前加载到固定内存
    )

    #加载验证数据集
    val_dataset = WaveEchoDataset(
        data_dir=modelConfig["val_data_dir"],  # <-- 使用 val_data_dir
        num_samples_per_file=8,
        num_rows_per_frame=16
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=modelConfig["batch_size"],  # 使用 modelConfig 中的配置
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    # 初始化模型
    net_model = UNet1D(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    # 加载预训练权重（如果指定）
    if modelConfig["train_load_weight"] is not None:
        weight_path =  modelConfig["train_load_weight"]
        try:
            net_model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"成功加载预训练权重：{weight_path}")
            logger.info(f"成功加载预训练权重：{weight_path}")
            start_epoch = 0
            #start_epoch = int(modelConfig["train_load_weight"].split('_')[1])
            #logger.info(f"将从epoch {start_epoch} 继续训练")
        except Exception as e:
            logger.error(f"加载预训练权重失败：{e}")
            logger.info("将从头开始训练...")
            start_epoch = 0
    else:
        start_epoch = 0
        logger.info("从头开始训练...")

    # 初始化训练器
    trainer = GaussianDiffusionTrainer1DReg(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
        modelConfig["classify_T"]
    ).to(device)

    # 初始化优化器
    optimizer = torch.optim.AdamW(
        trainer.fc.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )

    # 设置学习率调度器
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0,
        last_epoch=-1
    )
    
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )

    #冻结预训练的DDPM模型参数，只训练分类头部分
    print("Freeze DDPM model parameters...")
    for child in list(net_model.children()):#遍历模型的所有子模块
        for param in child.parameters():
            param.requires_grad = False# 将参数的requires_grad属性设置为False，即不计算梯度，不更新参数

    # 创建保存目录
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    
    # 创建训练损失图表保存目录
    loss_plot_dir = os.path.join(modelConfig["save_weight_dir"], "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)

    # 开始训练，初始化全局步数计数器
    global_step = 0 # 开始训练循环，从start_epoch开始，到modelConfig["epoch"]结束
    for e in range(start_epoch, modelConfig["epoch"]):
        net_model.train() # 将模型设置为训练模式，启用Dropout等训练时才使用的层
        trainer.fc.train()# 将训练器的全连接层设置为训练模式

        # 初始化用于计算 epoch 批次平均 MSE 的变量
        epoch_loss_sum = 0.0
        num_batches = 0# 初始化epoch内的批次数

        # 每个epoch开始时清理内存，释放不必要的内存
        torch.cuda.empty_cache()
        # 强制进行垃圾回收，释放不再使用的对象占用的内存
        gc.collect()

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            tqdmDataLoader.set_description(f"Epoch {e+1}/{modelConfig['epoch']}")
            # 遍历数据加载器中的每个批次
            for batch_idx, (data, heights, stamps) in enumerate(tqdmDataLoader):
                # 重塑数据以适应模型
                B = data.shape[0]
                num_samples_in_batch = B * 128# 计算当前批次中的总样本数量
                x_0 = data.view(num_samples_in_batch, 1, 128).to(device)# 重塑数据的形状，使其适应模型的输入要求，并将数据移动到指定的设备上
                heights = heights.view(num_samples_in_batch).to(device) #重塑波高数据的形状，并将其移动到指定的设备上
                
                # 计算当前批次的损失
                current_batch_losses = trainer(x_0, heights)
                
                # 计算当前批次的平均损失
                batch_avg_loss = current_batch_losses.mean()

                # 累加批次平均损失
                epoch_loss_sum += batch_avg_loss.item()
                # 累加批次数
                num_batches += 1
                
                # 计算用于梯度累积的损失 (使用批次平均损失)，将批次平均损失乘以1000并除以梯度累积步数
                loss = batch_avg_loss * 1000. / accumulation_steps

                # 反向传播，计算梯度
                loss.backward()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 对训练器全连接层的参数进行梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        trainer.fc.parameters(),
                        modelConfig["grad_clip"]
                    )
                    # 更新模型参数
                    optimizer.step()
                    # 清空梯度，准备下一次反向传播
                    optimizer.zero_grad()
                
                # 获取当前的学习率
                current_lr = optimizer.state_dict()['param_groups'][0]["lr"]
                tqdmDataLoader.set_postfix(ordered_dict={
                    "batch_loss": f"{batch_avg_loss.item():.4f}",
                    "LR": f"{current_lr:.6f}"
                })
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

                global_step += 1

        # 每个epoch结束时清理内存
        torch.cuda.empty_cache()
        gc.collect()

        # 计算整个训练集的批次平均 MSE
        avg_train_mse = epoch_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_mse)  # 记录每个 epoch 的批次平均 MSE

        ### eval (使用相同的批次平均法)
        net_model.eval()# 将模型设置为评估模式，禁用Dropout等训练时才使用的层
        trainer.fc.eval()# 将训练器的全连接层设置为评估模式
        val_loss_sum = 0.0  # 初始化验证损失总和
        num_val_batches = 0  # 初始化验证批次数
        with torch.no_grad(): # 禁用梯度计算，减少内存消耗，提高验证速度
            with tqdm(val_dataloader, dynamic_ncols=True, desc=f"Validation Epoch {e + 1}") as tqdmValDataLoader:
                # 遍历验证数据加载器中的每个批次
                for data, heights, stamps in tqdmValDataLoader:
                    B = data.shape[0]# 获取当前批次的样本数量
                    num_samples_in_batch = B * 128# 计算当前批次中的总样本数量
                    x_0 = data.view(num_samples_in_batch, 1, 128).to(device)# 重塑数据的形状，使其适应模型的输入要求，并将数据移动到指定的设备上
                    heights = heights.view(num_samples_in_batch).to(device)# 重塑波高数据的形状，并将其移动到指定的设备上

                    # 计算当前批次的损失
                    current_batch_losses = trainer(x_0, heights)

                    # 计算当前验证批次的平均损失
                    val_batch_avg_loss = current_batch_losses.mean().item()

                    # 累加验证批次平均损失
                    val_loss_sum += val_batch_avg_loss
                    # 累加验证批次数
                    num_val_batches += 1

                    # 更新进度条后缀
                    tqdmValDataLoader.set_postfix(ordered_dict={
                        "val_batch_loss": f"{val_batch_avg_loss:.4f}"
                    })

        # 计算整个验证集的批次平均 MSE
        avg_val_mse = val_loss_sum / num_val_batches if num_val_batches > 0 else 0

        # 使用批次平均 MSE 进行打印和后续处理
        print("Epoch: %d, train_mse:%.4f val_mse:%.4f" % (e, avg_train_mse, avg_val_mse))  # 打印两个批次平均 MSE

        # 早停检查 - 使用批次平均验证 MSE
        if avg_val_mse < best_loss:
            best_loss = avg_val_mse  # 更新 best_loss 为平均 MSE
            patience_counter = 0
            # 保存最佳模型 - 保存完整模型（预加载的扩散模型 + 训练好的分类头）
            best_model_path = os.path.join(modelConfig["save_weight_dir"], 'best_model_reg.pt')
            # 修改保存方式，确保与推理代码一致
            model_state_dict = {
                f"0.{k}": v for k, v in net_model.state_dict().items()
            }
            fc_state_dict = {
                f"1.{k}": v for k, v in trainer.fc.state_dict().items()
            }
            # 合并两个字典
            full_state_dict = {**model_state_dict, **fc_state_dict}
            torch.save(full_state_dict, best_model_path)
            log_msg = f"保存最佳回归层模型，验证 MSE：{best_loss:.4f}"
            logger.info(log_msg)
            print(log_msg)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_msg = f"早停触发！连续{patience}个epoch没有改善。"
                logger.info(log_msg)
                print(log_msg)
                break

        # 保存检查点 - 保存完整模型（预加载的扩散模型 + 训练好的分类头）
        save_path = os.path.join(modelConfig["save_weight_dir"], f'ckpt_reg_{e}_.pt')
        try:
            # 修改保存方式，确保与推理代码一致
            model_state_dict = {
                f"0.{k}": v for k, v in net_model.state_dict().items()
            }
            fc_state_dict = {
                f"1.{k}": v for k, v in trainer.fc.state_dict().items()
            }
            # 合并两个字典
            full_state_dict = {**model_state_dict, **fc_state_dict}
            torch.save(full_state_dict, save_path)
            logger.info(f"Epoch {e}: Checkpoint saved to {save_path}")
        except Exception as save_error:
            logger.error(f"Epoch {e}: Failed to save checkpoint: {save_error}")

        # 更新学习率 (保持不变)
        warmUpScheduler.step()

        # 结束wandb运行
        # if run:
        #     try:
        #         wandb.finish()
        #         logger.info("Weights & Biases run finished.")
        #     except Exception as finish_e:
        #         logger.error(f"Error finishing Weights & Biases run: {finish_e}")

if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 添加测试代码
    logger.info("测试数据集加载...")
    test_dataset = WaveEchoDataset(
        data_dir="./train_data/*.npy",
        num_samples_per_file=8,
        num_rows_per_frame=16
    )

    # 测试第一个样本
    test_data, test_heights, test_stamps = test_dataset[0]
    logger.info(f"数据形状: {test_data.shape}")
    logger.info(f"波高形状: {test_heights.shape}")
    logger.info(f"时间戳: {test_stamps}")

    # 原有的modelConfig配置
    modelConfig = {
        "data_dir": "./train_data/*.npy",
        "state": "train",  # or eval
        "epoch": 20,  # 200
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "seq_length": 128,
        "num_samples": 10000,
        "grad_clip": 1.,
        "device": "cuda:0",  # MAKE SURE YOU HAVE A GPU
        "train_load_weight": None,
        "save_weight_dir": "./Checkpoints_1d/",
        "sampled_dir": "./SampledImgs_1d/",
        "nrow": 8,
        "sample_interval": 10,  # Sample every 10 epochs
        "classify_T": 100  # Added for train_reg function
    }

    # 添加额外的数据验证
    logger.info("\n验证数据目录...")
    import glob

    files = glob.glob(modelConfig["data_dir"])
    if not files:
        logger.warning(f"警告：在指定路径下没有找到.npy文件！")
        logger.warning(f"请检查路径是否正确：{modelConfig['data_dir']}")
    else:
        logger.info(f"找到{len(files)}个.npy文件")
        logger.info(f"第一个文件：{files[0]}")

    logger.info("\n开始训练...")
    train(modelConfig)
