from Diffusion.Train import train_classify


def main(model_config = None):
    modelConfig = {
        # Training mode: "train" for training, "eval" for evaluation
        "state": "train",
        
        # Training hyperparameters
        "epoch": 200,          # Total number of training epochs
        "batch_size": 32,      # Number of samples per batch
        "lr": 1e-4,            # Learning rate for optimizer
        "multiplier": 2.,      # Learning rate warmup multiplier
        "grad_clip": 1.,       # Maximum norm for gradient clipping
        
        # Diffusion process parameters
        "T": 1000,             # Total number of diffusion timesteps
        "beta_1": 1e-4,        # Starting value of beta schedule
        "beta_T": 0.02,        # Ending value of beta schedule
        
        # Model architecture parameters
        "channel": 128,                # Base number of channels in U-Net
        "channel_mult": [1, 2, 3, 4],  # Channel multipliers for different scales
        "attn": [2],                   # Positions to add attention blocks
        "num_res_blocks": 2,           # Number of residual blocks per scale
        "dropout": 0.15,               # Dropout probability
        "img_size": 32,                # Input image size (height and width)
        
        # Hardware settings
        "device": "cpu",#"cuda:0",            # Device to use for training (GPU)
        
        # Model checkpoint settings
        "training_load_weight": None,#"./Checkpoints/ckpt_199_.pt",  # Path to load pre-trained weights
        "save_weight_dir": "./Checkpoints/",                   # Directory to save model checkpoints
        "test_load_weight": "ckpt_199_.pt",                    # Path to load weights for evaluation
        
        # Output image settings
        "sampled_dir": "./SampledImgs/",                       # Directory to save generated images
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",      # Filename for noisy intermediate images
        "sampledImgName": "SampledNoGuidenceImgs.png",         # Filename for final generated images
        "nrow": 8,                                             # Number of images per row in output grid

        # Classification-specific parameters
        "classify_T": 30,              # Specific timestep to use for classification
        "head": 'fc',                  # Type of classification head: 'fc' (Fully Connected), 
                                      # 'mlp' (Multi-Layer Perceptron), 
                                      # 'attn' (Attention-based), 
                                      # 'cnn' (Convolutional Neural Network)
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train_classify(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
