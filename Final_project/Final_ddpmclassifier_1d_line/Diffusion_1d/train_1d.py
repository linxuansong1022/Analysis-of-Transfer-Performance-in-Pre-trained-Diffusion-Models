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
        net_model.parameters(),
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

    # 初始化训练器
    trainer = GaussianDiffusionTrainer1D(
        net_model,
        modelConfig["beta_1"],
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
        epoch_loss = 0.0
        num_batches = 0
        
        # 每个epoch开始时清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            tqdmDataLoader.set_description(f"Epoch {e+1}/{modelConfig['epoch']}")
            
            for batch_idx, (data, heights, stamps) in enumerate(tqdmDataLoader):
                # 重塑数据以适应模型
                B = data.shape[0]
                x_0 = data.view(B*128, 1, 128).to(device)
                heights = heights.view(B*128).to(device)
                
                # 计算损失并进行梯度累积
                current_loss = trainer(x_0).mean()
                loss = current_loss * 1000. / accumulation_steps  # 除以累积步数

                # 反向传播
                loss.backward()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(),
                        modelConfig["grad_clip"]
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 更新进度条
                current_lr = optimizer.state_dict()['param_groups'][0]["lr"]
                epoch_loss += current_loss.item()
                num_batches += 1
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
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
        train_losses.append(avg_epoch_loss)
        
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
    


    # 加载数据集
    dataset = WaveEchoDataset(
        data_dir=modelConfig["data_dir"],
        num_samples_per_file=8,
        num_rows_per_frame=16
    )
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],  # 使用 modelConfig 中的配置
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )


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
        modelConfig["T"]
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

    
    # 冻结预训练的 DDPM 模型参数
    print("Freeze DDPM model parameters...")
    for child in list(net_model.children()):
        for param in child.parameters():
            param.requires_grad = False

    # 创建保存目录
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
    
    # 创建训练损失图表保存目录
    loss_plot_dir = os.path.join(modelConfig["save_weight_dir"], "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)

    # 开始训练
    global_step = 0
    for e in range(start_epoch, modelConfig["epoch"]):
        net_model.train()
        trainer.fc.train()

        # 初始化用于计算 epoch 批次平均 MSE 的变量
        epoch_loss_sum = 0.0
        num_batches = 0

        # 每个epoch开始时清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            tqdmDataLoader.set_description(f"Epoch {e+1}/{modelConfig['epoch']}")
            
            for batch_idx, (data, heights, stamps) in enumerate(tqdmDataLoader):
                # 重塑数据以适应模型
                B = data.shape[0]
                num_samples_in_batch = B * 128
                x_0 = data.view(num_samples_in_batch, 1, 128).to(device)
                heights = heights.view(num_samples_in_batch).to(device)
                
                # 计算当前批次的损失
                current_batch_losses = trainer(x_0, heights)
                
                # 计算当前批次的平均损失
                batch_avg_loss = current_batch_losses.mean()

                # 累加批次平均损失
                epoch_loss_sum += batch_avg_loss.item()
                num_batches += 1
                
                # 计算用于梯度累积的损失 (使用批次平均损失)
                loss = batch_avg_loss * 1000. / accumulation_steps

                # 反向传播
                loss.backward()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainer.fc.parameters(),
                        modelConfig["grad_clip"]
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 更新进度条 (显示当前批次的平均损失)
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

        # 计算整个训练 epoch 的批次平均 MSE
        avg_train_mse = epoch_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_mse) # 记录每个 epoch 的批次平均 MSE

        ### eval (使用相同的批次平均法)
        net_model.eval()
        trainer.fc.eval()
        val_loss_sum = 0.0 # 累加验证批次的平均损失
        num_val_batches = 0 # 累加验证批次数
        with torch.no_grad():
            with tqdm(val_dataloader, dynamic_ncols=True, desc=f"Validation Epoch {e+1}") as tqdmValDataLoader:
                for data, heights, stamps in tqdmValDataLoader:
                    B = data.shape[0]
                    num_samples_in_batch = B * 128
                    x_0 = data.view(num_samples_in_batch, 1, 128).to(device)
                    heights = heights.view(num_samples_in_batch).to(device)

                    # 计算当前批次的损失
                    current_batch_losses = trainer(x_0, heights)
                    
                    # 计算当前验证批次的平均损失
                    val_batch_avg_loss = current_batch_losses.mean().item()
                    
                    # 累加验证批次平均损失和批次数
                    val_loss_sum += val_batch_avg_loss
                    num_val_batches += 1
                    
                    # 更新进度条后缀
                    tqdmValDataLoader.set_postfix(ordered_dict={
                        "val_batch_loss": f"{val_batch_avg_loss:.4f}"
                    })

        # 计算整个验证集的批次平均 MSE
        avg_val_mse = val_loss_sum / num_val_batches if num_val_batches > 0 else 0

        # 使用批次平均 MSE 进行打印和后续处理
        print("Epoch: %d, train_mse:%.4f val_mse:%.4f" % (e, avg_train_mse, avg_val_mse)) # 打印两个批次平均 MSE

        # 早停检查 - 使用批次平均验证 MSE
        if avg_val_mse < best_loss:
            best_loss = avg_val_mse # 更新 best_loss 为平均 MSE
            patience_counter = 0
            # 保存最佳模型 - 建议只保存回归头 trainer.fc 的状态
            best_model_path = os.path.join(modelConfig["save_weight_dir"], 'best_model_reg.pt')
            torch.save(trainer.fc.state_dict(), best_model_path) # <-- 保存 trainer.fc.state_dict()
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

        # 绘制训练损失曲线 (现在是真实的平均MSE)
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training MSE') # 更新标签
        plt.title('Training MSE Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MSE') # 更新 Y 轴标签
        plt.legend()
        plt.grid(True)
        
        # 保存损失曲线图
        loss_plot_path = os.path.join(loss_plot_dir, f'loss_plot_epoch_{e+1}.png')
        plt.savefig(loss_plot_path)
        plt.close()

        

        # 保存检查点 - 同样建议只保存回归头的状态
        save_path = os.path.join(modelConfig["save_weight_dir"], f'ckpt_reg_{e}_.pt') # 建议添加 reg 区分
        try:
            torch.save(trainer.fc.state_dict(), save_path) # <-- 保存 trainer.fc.state_dict()
            logger.info(f"Epoch {e}: Checkpoint saved to {save_path}")
        except Exception as save_error:
            logger.error(f"Epoch {e}: Failed to save checkpoint: {save_error}")

        # 定期生成样本 (这部分与回归任务关系不大，但保持不变)
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


                plt.close(fig)

            net_model.train()
        # 更新学习率 (保持不变)
        warmUpScheduler.step()

    # 保存最终的训练损失曲线 (现在是真实的平均MSE)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training MSE') # 更新标签
    plt.title('Final Training MSE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE') # 更新 Y 轴标签
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
        "sample_interval": 10  # Sample every 10 epochs
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
