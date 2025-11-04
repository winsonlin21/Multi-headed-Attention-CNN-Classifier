import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 42
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import datasets and create dataloaders
# returns training dataset (train_loader), validation dataset (val_loader), test dataset (test_loader), 
# input channels (in_channels), image size (image_size), number of classes (num_classes
def get_dataset(name, batch_size=64):
    if name == "MNIST":
        mean, std = (0.1307,), (0.3081,)
        in_channels, image_size, num_classes = 1, 28, 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.MNIST

    elif name == "FashionMNIST":
        mean, std = (0.2860,), (0.3530,)
        in_channels, image_size, num_classes = 1, 28, 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.FashionMNIST

    elif name == "CIFAR10":
        mean = [0.4914, 0.4822, 0.4465]
        std =  [0.2470, 0.2435, 0.2616]
        in_channels, image_size, num_classes = 3, 32, 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.CIFAR10

    elif name == "CIFAR100":
        mean = [0.5071, 0.4867, 0.4408]
        std =  [0.2675, 0.2565, 0.2761]
        in_channels, image_size, num_classes = 3, 32, 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.CIFAR100

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # loads dataset 
    train_raw = dataset_cls(root="./data", train=True, download=True, transform=transform)
    test_raw  = dataset_cls(root="./data", train=False, download=True, transform=transform)
    # splits into train and val
    train_data, val_data = random_split(train_raw, [int(0.9 * len(train_raw)), len(train_raw) - int(0.9 * len(train_raw))],
                                        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size)
    test_loader  = DataLoader(test_raw, batch_size=batch_size)

    return train_loader, val_loader, test_loader, in_channels, image_size, num_classes

#trains the model for one epoch and evaluates on validation set afterwards
#returns training loss(avg_loss), training accuracy(accuracy), validation loss (val_loss), validation accuracy (val_acc)
def train_one_epoch(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    return avg_loss, accuracy, val_loss, val_acc

# evaluates the model on the given dataloader
# returns average loss (avg_loss) and accuracy (accuracy)
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

class BaseCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # normalizes each channel it helps training be more stable
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # reduces the spatial size  This lowers computation and gives some translation invariance.
        self.pool = nn.MaxPool2d(2, 2)

        # randomly zeroes whole channels  prevents overfitting
        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.dropout(x)
        return x

class AttentionCNN(nn.Module):
    def __init__(self, base_cnn, num_features=256, num_classes=10, num_heads=4):
        super().__init__()

        self.base = base_cnn
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads
        
        assert num_features % num_heads == 0, "num_features must be divisible by num_heads"
        
        # multiple attention heads
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=1)
            ) for _ in range(num_heads)
        ])
        
        self.head_projections = nn.ModuleList([
            nn.Conv2d(256, self.head_dim, kernel_size=1) 
            for _ in range(num_heads)
        ])
        
        self.fc1 = nn.Linear(num_features, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)
    
    def forward(self, x):
        feats = self.base(x)
        B, C, H, W = feats.shape
        
        head_outputs = []
        
        for i in range(self.num_heads):
            projected_feats = self.head_projections[i](feats)
            
            attn_logits = self.attn_heads[i](feats)
            attn_logits = attn_logits.view(B, -1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            
            flat_projected = projected_feats.view(B, self.head_dim, H * W)
            weighted_feats = flat_projected * attn_weights.unsqueeze(1)
            head_output = weighted_feats.sum(dim=2)
            
            head_outputs.append(head_output)
        
        agg = torch.cat(head_outputs, dim=1)
        
        x = F.relu(self.fc1(agg))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)
    
if __name__ == '__main__':


    # "MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"
    train_loader, val_loader, test_loader, in_channels, image_size, num_classes = get_dataset("MNIST", batch_size=64)

    criterion = nn.CrossEntropyLoss()

    base_model = BaseCNN(in_channels=in_channels).to(device)

    print("Training Attention-based Model...")

    att_model = AttentionCNN(base_model, 256, num_classes=num_classes).to(device)
    att_optimizer = torch.optim.Adam(att_model.parameters(), lr=0.001)

    att_tr_loss = []
    att_val_acc = []

    for epoch in range(1, 11):
        tr_loss, tr_acc, val_loss, val_acc = train_one_epoch(att_model, train_loader, val_loader, att_optimizer, criterion, device)
        att_tr_loss.append(tr_loss)
        att_val_acc.append(val_acc)
        print(f"[Att] Epoch {epoch}: Train Acc = {tr_acc:.4f}, Val Acc = {val_acc:.4f}")

        # Save checkpoints: last and best (by validation accuracy)
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        def save_checkpoint(model, optimizer, epoch, val_acc, path):
            """Save model and optimizer state to `path`."""
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'att_tr_loss': att_tr_loss,
                'att_val_acc': att_val_acc,
            }, path)

        # save last
        last_path = os.path.join(checkpoint_dir, "att_last.pt")
        save_checkpoint(att_model, att_optimizer, epoch, val_acc, last_path)

        # save best
        best_path = os.path.join(checkpoint_dir, "att_best.pt")
        # if best doesn't exist or this val_acc is better, overwrite
        if not os.path.exists(best_path):
            save_checkpoint(att_model, att_optimizer, epoch, val_acc, best_path)
            best_val = val_acc
        else:
            # load recorded best val_acc without loading whole model
            try:
                best_ckpt = torch.load(best_path, map_location=device)
                best_val = best_ckpt.get('val_acc', -1)
            except Exception:
                best_val = -1

            if val_acc >= best_val:
                save_checkpoint(att_model, att_optimizer, epoch, val_acc, best_path)

    # after training finished, evaluate using the best checkpoint if available

    test_loss, test_acc = evaluate(att_model, test_loader, criterion, device)
    print(f"[Att] Test Accuracy = {test_acc:.4f}")