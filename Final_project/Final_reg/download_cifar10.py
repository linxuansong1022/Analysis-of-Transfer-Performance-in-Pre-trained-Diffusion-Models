import torch
from torchvision import datasets, transforms

def download_cifar10():
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load the training set
    print("Downloading CIFAR-10 training set...")
    train_dataset = datasets.CIFAR10(
        root='./CIFAR10',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load the test set
    print("Downloading CIFAR-10 test set...")
    test_dataset = datasets.CIFAR10(
        root='./CIFAR10',
        train=False,
        download=True,
        transform=transform
    )
    
    print("CIFAR-10 dataset downloaded successfully!")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

if __name__ == '__main__':
    download_cifar10() 
    