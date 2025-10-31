from Diffusion_1d.train_1d import train_reg

def main(model_config=None):
    modelConfig = {
        "data_dir": "/root/autodl-tmp/WavePrediction/train_data/1/*.npy",
        "val_data_dir": "/root/autodl-tmp/WavePrediction/test_data/*.npy",
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size":10,
        "T": 1000,
        "channel": 64,
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
        "train_load_weight": "Checkpoints_1d/ckpt_20_.pt",
        "save_weight_dir": "./Checkpoints_1d_reg/",
        "sampled_dir": "./SampledImgs_1d_reg/",
        "nrow": 8,
        "sample_interval": 10  # Sample every 10 epochs
    }

    if model_config is not None:
        modelConfig.update(model_config)

    # 添加数据验证
    print("验证数据加载...")
    try:
        from Diffusion_1d.wave_echo_dataset import WaveEchoDataset
        test_dataset = WaveEchoDataset(
            data_dir=modelConfig["data_dir"],
            num_samples_per_file=8,
            num_rows_per_frame=16
        )
        test_data, test_heights, test_stamps = test_dataset[0]
        print(f"数据形状: {test_data.shape}")
        print(f"波高形状: {test_heights.shape}")
        print(f"时间戳: {test_stamps}")

        print("\n验证数据目录...")
        import glob
        files = glob.glob(modelConfig["data_dir"])
        if not files:
            print(f"警告：在指定路径下没有找到.npy文件！")
            print(f"请检查路径是否正确：{modelConfig['data_dir']}")
            return
        print(f"找到{len(files)}个.npy文件")
        print(f"第一个文件：{files[0]}")

    except Exception as e:
        print(f"数据加载验证失败：{str(e)}")
        return

    print("\n开始训练...")
    if modelConfig["state"] == "train":
        train_reg(modelConfig)
    else:
        print("暂不支持评估模式")


if __name__ == '__main__':
    main()