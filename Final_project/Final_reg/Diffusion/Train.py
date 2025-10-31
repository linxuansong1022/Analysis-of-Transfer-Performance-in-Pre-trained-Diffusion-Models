
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

from Diffusion import GaussianDiffusionTrainer,GaussianDiffusionClassifyTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


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
    print("train val:", len(train_dataloader), len(val_dataloader))

    # 模型设置：构建 DDPM 模型，并加载预训练权重
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(modelConfig["training_load_weight"], map_location=device))
        print("Load pretrain")

    # 构建分类器，封装在 GaussianDiffusionClassifyTrainer 中
    trainer = GaussianDiffusionClassifyTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        modelConfig["head"], modelConfig["classify_T"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        trainer.fc.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    # 冻结预训练的 DDPM 模型参数
    print("Freeze DDPM model parameters...")
    for child in list(net_model.children()):
        for param in child.parameters():
            param.requires_grad = False

    # 定义打印参数量的辅助函数
    def print_params(model, name):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name} 参数量: {total_params}")

    # 打印预训练模型和分类器的参数量
    print_params(net_model, "预训练DDPM模型")
    print_params(trainer.fc, "分类器")

    # 开始训练
    for e in range(modelConfig["epoch"]):
        net_model.train()
        trainer.fc.train()
        correct = 0
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                pre, loss = trainer(x_0, labels)
                loss = loss.sum() / 1000.
                loss.backward()

                pred = nn.Softmax(dim=1)(pre)
                pred = pred.max(1, keepdim=True)[1]  # 获取预测类别
                correct += pred.eq(labels.view_as(pred)).sum().item()

                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(),
                   os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

        train_acc = correct / len(train_dataloader.dataset)

        # 评估过程
        net_model.eval()
        trainer.fc.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                x_0 = images.to(device)
                labels = labels.to(device)
                pre, loss = trainer(x_0, labels)
                loss = loss.sum() / 1000.
                pred = nn.Softmax(dim=1)(pre)
                pred = pred.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
        val_acc = correct / len(val_dataloader.dataset)
        print("Epoch: %d, train_acc: %.4f, val_acc: %.4f" % (e, train_acc, val_acc))
