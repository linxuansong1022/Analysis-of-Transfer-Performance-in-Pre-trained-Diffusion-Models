from Diffusion.Train import train_classify


def main(model_config = None):
    modelConfig = {
        "state": "train",
        "epoch": 200,
        "batch_size": 32,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,#学习率预热调度器 (GradualWarmupScheduler) 的乘数因子。预热结束后，学习率会达到 初始学习率 * multiplier。
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1., #梯度裁剪的阈值。在优化器更新参数之前，如果参数梯度的范数超过 1.0，则将其缩放到 1.0，以防止梯度爆炸。
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": "./Checkpoints/ckpt_199_.pt",
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,

        "classify_T": 30,
        "head": 'attn', # fc / mlp / attn / cnn
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train_classify(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
