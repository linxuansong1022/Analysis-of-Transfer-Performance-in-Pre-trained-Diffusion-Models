# run_full_finetune_orchestrator.py

import torch
import torch.nn as nn # Needed for MSELoss potentially, though F.mse_loss is used
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import gc
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Assumed Imports from your project structure ---
# Make sure these modules are accessible in your Python path
try:
    from Diffusion_1d.wave_echo_dataset import WaveEchoDataset
    from Diffusion_1d.model_1d import UNet1D
    # Import the specific trainer and sampler needed
    from Diffusion_1d.diffusion_1d import GaussianDiffusionTrainer1DReg, GaussianDiffusionSampler1D
    # Import the scheduler if it's defined in a separate file (e.g., Scheduler.py)
    # If it's defined within train_1d.py, this needs adjustment or copy the class here.
    from Scheduler import GradualWarmupScheduler # Assuming Scheduler.py exists at the project root
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure that Diffusion_1d package and Scheduler.py are in your Python path.")
    print("If GradualWarmupScheduler is defined elsewhere, adjust the import path.")
    exit()
# --- Helper: extract function (often needed by diffusion trainers) ---
# It's safer to include it here in case it's not imported elsewhere
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v.to(device), index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# --- Main Fine-tuning Orchestration ---
def main():
    # --- Configuration ---
    modelConfig = {
        # Data paths (!!! PLEASE VERIFY/EDIT THESE PATHS !!!)
        "data_dir": "/root/autodl-tmp/WavePrediction/train_data/1/*.npy",
        "val_data_dir": "/root/autodl-tmp/WavePrediction/test_data/*.npy",
        "state": "train",
        "epoch": 200,
        "batch_size": 10,
        "T": 1000,
        "lr": 1e-5, # << Small learning rate for fine-tuning >>
        "multiplier": 1.,
        "grad_clip": 1.,
        "channel": 64, # Match pre-trained model's config
        "channel_mult": [1, 2, 3, 4], # Match pre-trained model's config
        "attn": [2], # Match pre-trained model's config
        "num_res_blocks": 2, # Match pre-trained model's config
        "dropout": 0.15, # Match pre-trained model's config
        "beta_1": 1e-4, # Match pre-trained model's config
        "beta_T": 0.02, # Match pre-trained model's config
        "seq_length": 128,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "train_load_weight": "Checkpoints_1d/ckpt_20_.pt", # << IMPORTANT: Path to pre-trained UNet weights >>
        "save_weight_dir": "./Checkpoints_1d_finetune/",
        "sampled_dir": "./SampledImgs_1d_finetune/",
        "nrow": 8,
        "sample_interval": 20
    }

    # --- Logging Setup ---
    log_dir = modelConfig["save_weight_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_finetune_orchestrator.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Fine-tuning Orchestrator Script ---")
    logger.info(f"Using device: {modelConfig['device']}")
    logger.info(f"Configuration: {modelConfig}")

    # --- Basic Path/File Validation ---
    valid_start = True
    if not glob.glob(modelConfig["data_dir"]):
        logger.error(f"错误：找不到训练数据文件！路径: {modelConfig['data_dir']}")
        valid_start = False
    if not glob.glob(modelConfig["val_data_dir"]):
        logger.error(f"错误：找不到验证数据文件！路径: {modelConfig['val_data_dir']}")
        valid_start = False
    if not modelConfig["train_load_weight"]:
        logger.warning("未指定预训练权重 'train_load_weight'，模型将从头训练 (非预期微调)。")
        # Set valid_start = True if training from scratch is ok
    elif not os.path.exists(modelConfig["train_load_weight"]):
        logger.error(f"错误：找不到预训练权重文件！路径: {modelConfig['train_load_weight']}")
        valid_start = False

    if not valid_start:
        logger.error("由于配置或文件路径问题，训练无法启动。")
        return # Exit if basic checks fail

    device = torch.device(modelConfig["device"])

    # --- Data Loading ---
    logger.info("加载数据集...")
    try:
        train_dataset = WaveEchoDataset(
            data_dir=modelConfig["data_dir"],
            num_samples_per_file=128, # Each item returns 128 sequences
            num_rows_per_frame=1
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=modelConfig["batch_size"], shuffle=True,
            num_workers=4, drop_last=True, pin_memory=True
        )
        val_dataset = WaveEchoDataset(
            data_dir=modelConfig["val_data_dir"],
            num_samples_per_file=128, num_rows_per_frame=1
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=modelConfig["batch_size"], shuffle=False,
            num_workers=4, drop_last=True, pin_memory=True
        )
        logger.info(f"训练集文件数: {len(train_dataset)}, 验证集文件数: {len(val_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             logger.error("错误：一个或多个数据集为空，请检查数据路径和内容。")
             return
    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return

    # --- Model Initialization & Loading Pre-trained Weights ---
    logger.info("初始化 UNet1D 模型...")
    net_model = UNet1D(
        T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    logger.info(f"尝试从 '{modelConfig['train_load_weight']}' 加载预训练权重...")
    if modelConfig["train_load_weight"] and os.path.exists(modelConfig["train_load_weight"]):
        try:
            pretrained_state_dict = torch.load(modelConfig["train_load_weight"], map_location=device)
            # Handle cases where weights might be saved inside a 'model' or 'trainer' key
            if 'model_state_dict' in pretrained_state_dict:
                 pretrained_state_dict = pretrained_state_dict['model_state_dict']
            elif 'net_model_state_dict' in pretrained_state_dict:
                 pretrained_state_dict = pretrained_state_dict['net_model_state_dict']
            elif 'trainer_state_dict' in pretrained_state_dict:
                 # Need to filter only UNet keys if loading from trainer state
                 unet_keys = net_model.state_dict().keys()
                 pretrained_state_dict = {k.replace('model.', ''): v for k, v in pretrained_state_dict['trainer_state_dict'].items() if k.replace('model.', '') in unet_keys}


            # Load into net_model, allowing missing/unexpected keys for flexibility
            missing_keys, unexpected_keys = net_model.load_state_dict(pretrained_state_dict, strict=False)

            logger.info(f"成功加载预训练权重。")
            if missing_keys:
                logger.warning(f"加载权重时缺少键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"加载权重时发现意外键: {unexpected_keys}")

        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}", exc_info=True)
            logger.warning("将从头开始训练模型...")
            # Optionally exit if pre-trained weights are mandatory:
            # return

    # --- Trainer Initialization (Wraps Model + Adds FC Head) ---
    logger.info("初始化 GaussianDiffusionTrainer1DReg...")
    trainer = GaussianDiffusionTrainer1DReg(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
    ).to(device)

    # --- Optimizer and Scheduler ---
    logger.info(f"设置 AdamW 优化器，初始学习率: {modelConfig['lr']}")
    optimizer = torch.optim.AdamW(
        trainer.parameters(), # Optimize all parameters of the trainer
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )

    logger.info("设置学习率调度器 (Warmup + Cosine)...")
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"] - (modelConfig["epoch"] // 10), # Adjust T_max for warmup
        eta_min=1e-7 # Smaller minimum LR
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig.get("multiplier", 1),
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )

    # --- Training Loop ---
    logger.info("开始 Fine-tuning 训练循环...")
    accumulation_steps = 4 # Gradient accumulation steps
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    start_epoch = 0 # Assuming starting from epoch 0 for fine-tuning

    for e in range(start_epoch, modelConfig["epoch"]):
        trainer.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        torch.cuda.empty_cache()
        gc.collect()

        train_pbar = tqdm(train_dataloader, desc=f"Fine-tuning Epoch {e+1}/{modelConfig['epoch']}", dynamic_ncols=True)
        optimizer.zero_grad()

        for batch_idx, (data_batch, heights_batch, _) in enumerate(train_pbar):
            B, InnerB, _, L = data_batch.shape
            x_0 = data_batch.view(B * InnerB, 1, L).to(device)
            heights = heights_batch.view(B * InnerB).to(device)

            current_batch_losses = trainer(x_0, heights) # Forward pass
            batch_avg_loss = current_batch_losses.mean()

            loss = (batch_avg_loss / accumulation_steps) * 1000. # Scale loss

            loss.backward() # Calculate gradients

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), modelConfig["grad_clip"])
                optimizer.step() # Update weights
                optimizer.zero_grad() # Reset gradients

            epoch_loss_sum += batch_avg_loss.item()
            num_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix(ordered_dict={
                "batch_loss": f"{batch_avg_loss.item():.5f}",
                "LR": f"{current_lr:.6e}"
            })

        warmUpScheduler.step() # Step scheduler after epoch
        avg_train_mse = epoch_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_mse)

        # --- Validation ---
        trainer.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        torch.cuda.empty_cache()
        gc.collect()
        val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {e+1}", dynamic_ncols=True)
        with torch.no_grad():
            for data_batch, heights_batch, _ in val_pbar:
                B, InnerB, _, L = data_batch.shape
                x_0 = data_batch.view(B * InnerB, 1, L).to(device)
                heights = heights_batch.view(B * InnerB).to(device)
                val_losses_batch = trainer(x_0, heights)
                val_batch_avg_loss = val_losses_batch.mean().item()
                val_loss_sum += val_batch_avg_loss
                num_val_batches += 1
                val_pbar.set_postfix(ordered_dict={"val_batch_loss": f"{val_batch_avg_loss:.5f}"})

        avg_val_mse = val_loss_sum / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_mse)
        logger.info(f"Epoch: {e+1:03d}, Train_MSE: {avg_train_mse:.5f}, Val_MSE: {avg_val_mse:.5f}, LR: {current_lr:.6e}")

        # --- Model Saving & Early Stopping ---
        if avg_val_mse < best_loss:
            best_loss = avg_val_mse
            patience_counter = 0
            best_model_path = os.path.join(modelConfig["save_weight_dir"], 'best_model_finetune.pt')
            try:
                # Save the entire trainer state (includes fine-tuned UNet and fc head)
                torch.save(trainer.state_dict(), best_model_path)
                logger.info(f"保存最佳微调模型 (Epoch {e+1}), 验证 MSE: {best_loss:.5f}")
            except Exception as save_e:
                logger.error(f"保存最佳模型失败: {save_e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！连续 {patience} 个 epoch 没有改善。最佳 Val MSE: {best_loss:.5f}")
                break

        # --- Plotting ---
        loss_plot_dir = os.path.join(modelConfig["save_weight_dir"], "loss_plots_finetune")
        os.makedirs(loss_plot_dir, exist_ok=True)
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training MSE')
            plt.plot(val_losses, label='Validation MSE')
            plt.title(f'Fine-tuning MSE (Epoch {e+1})')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.grid(True)
            if val_losses: # Add marker for best validation loss
                 min_val_idx = np.argmin(val_losses)
                 plt.scatter(min_val_idx, val_losses[min_val_idx], color='red', zorder=5, label=f'Best Val MSE: {val_losses[min_val_idx]:.5f} (Epoch {min_val_idx+1})')
            plt.legend()
            plt.savefig(os.path.join(loss_plot_dir, f'loss_plot_epoch_{e+1}.png'))
            plt.close()
        except Exception as plot_e:
             logger.error(f"绘制损失图失败: {plot_e}")

        # --- Save Checkpoint ---
        ckpt_path = os.path.join(modelConfig["save_weight_dir"], f'ckpt_finetune_{e+1:03d}.pt')
        try:
            torch.save({
                'epoch': e + 1,
                'trainer_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': warmUpScheduler.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, ckpt_path)
            # logger.info(f"Epoch {e+1}: Checkpoint saved.") # Reduce log verbosity
        except Exception as save_error:
            logger.error(f"Epoch {e+1}: Failed to save checkpoint: {save_error}")

        # --- Optional Sampling (can be commented out if not needed) ---
        if (e + 1) % modelConfig["sample_interval"] == 0:
             logger.info(f"Epoch {e+1}: Generating samples...")
             trainer.eval()
             sampler = GaussianDiffusionSampler1D(
                 trainer.model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]
             ).to(device)
             try:
                 with torch.no_grad():
                     x_T = torch.randn([modelConfig["nrow"], 1, modelConfig["seq_length"]], device=device)
                     x_0 = sampler(x_T)
                     # Plotting samples... (same as before)
                     fig = plt.figure(figsize=(10, 10)); fig.suptitle(f"Epoch {e+1} Samples")
                     for i in range(modelConfig["nrow"]):
                         ax = fig.add_subplot(modelConfig["nrow"] // 2, 2, i + 1)
                         ax.plot(x_0[i, 0].cpu().numpy()); ax.set_ylim(-1, 1); ax.set_title(f"Sample {i+1}")
                     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                     plt.savefig(os.path.join(modelConfig["sampled_dir"], f"sample_finetune_{e+1:03d}.png"))
                     plt.close(fig)
             except Exception as sample_e:
                  logger.error(f"Epoch {e+1}: Failed during sample generation: {sample_e}")


    # --- Final Actions ---
    logger.info("微调训练完成！ Best Validation MSE: {:.5f}".format(best_loss))
    # Save final plot
    loss_plot_dir = os.path.join(modelConfig["save_weight_dir"], "loss_plots_finetune")
    final_plot_path = os.path.join(loss_plot_dir, 'final_loss_plot.png')
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training MSE')
        plt.plot(val_losses, label='Validation MSE')
        plt.title('Final Fine-tuning MSE Curve')
        plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
        if val_losses:
             min_val_idx = np.argmin(val_losses)
             plt.scatter(min_val_idx, val_losses[min_val_idx], color='red', zorder=5, label=f'Best Val MSE: {val_losses[min_val_idx]:.5f} (Epoch {min_val_idx+1})')
             plt.legend()
        plt.savefig(final_plot_path)
        plt.close()
        logger.info(f"Final loss plot saved to {final_plot_path}")
    except Exception as plot_e:
         logger.error(f"保存最终损失图失败: {plot_e}")

    logger.info("--- Fine-tuning Orchestrator Script Finished ---")


if __name__ == '__main__':
    # Ensure matplotlib uses a non-interactive backend if running on servers without GUI
    try:
        plt.switch_backend('Agg')
        print("Matplotlib backend set to Agg.")
    except ImportError:
        print("Could not import Agg backend for Matplotlib.")

    main()