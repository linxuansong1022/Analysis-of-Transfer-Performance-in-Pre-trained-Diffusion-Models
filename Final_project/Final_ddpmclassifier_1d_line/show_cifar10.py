import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def show_cifar10_images():
    # Define the transform (without normalization for visualization)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the dataset
    dataset = datasets.CIFAR10(
        root='./CIFAR10',
        train=True,
        download=False,  # We already downloaded it
        transform=transform
    )
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    
    # Show images in a grid
    for i in range(16):
        plt.subplot(4, 4, i+1)
        # Convert tensor to numpy array and transpose to (H, W, C)
        img = images[i].numpy().transpose((1, 2, 0))
        # Clip values to [0, 1] for proper display
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_cifar10_images() 