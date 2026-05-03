import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32):
    print("loading data set...")
    base_path = kagglehub.dataset_download("lantian773030/pokemonclassification")
    
    dataset_path = base_path
    for root, dirs, files in os.walk(base_path):
        if len(dirs) > 100:
            dataset_path = root
            break
            
    print(f"dataset path: {dataset_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    print(f"pokemon classes: {len(full_dataset.classes)}") 
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes