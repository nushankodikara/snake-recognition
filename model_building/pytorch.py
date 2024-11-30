import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 16
IMAGE_DIM = 480

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SnakeDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            train (bool): Whether this is training or test data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_file)
        self.train = train
        self.transform = transform
        self.root_dir = "../input/165-different-snakes-species"
        
        # Create class mapping
        self.classes = sorted(self.data['class_id'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        try:
            # Construct image path using class_id and UUID
            split = 'train' if self.train else 'test'
            class_id = str(self.data.iloc[idx]['class_id'])
            uuid = self.data.iloc[idx]['UUID']
            img_path = os.path.join(self.root_dir, split, class_id, f"{uuid}.jpg")
            
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get class label
            label = self.class_to_idx[self.data.iloc[idx]['class_id']]
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # Return a default/placeholder image and label
            placeholder = torch.zeros((3, IMAGE_DIM, IMAGE_DIM))
            return placeholder, 0

# Initialize datasets and dataloaders
train_dataset = SnakeDataset(
    csv_file="../input/165-different-snakes-species/Csv/train.csv",
    train=True,
    transform=transform
)

test_dataset = SnakeDataset(
    csv_file="../input/165-different-snakes-species/Csv/test.csv",
    train=False,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# Initialize model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

def train_model():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training')
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def validate_model():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

# Training loop
print("Starting training...")
best_acc = 0
for epoch in range(EPOCHS):
    print(f'\nEpoch [{epoch+1}/{EPOCHS}]')
    train_loss, train_acc = train_model()
    val_loss, val_acc = validate_model()
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, 'best_snake_classifier.pth')

print("Training completed!")
print(f"Best validation accuracy: {best_acc:.2f}%")
