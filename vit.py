import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

BATCH_SIZE = 512
EPOCHS = 50

# Check if GPU is available
# if (torch.cuda.is_available()):
#     print("Using GPU")
# else:
#     print("Using CPU")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Chargement et affichage d'une image du dataset CIFAR-10
def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    return next(iter(trainloader))

# image, label = load_cifar10()
# plt.imshow(image.squeeze().permute(1, 2, 0))
# plt.show()

# 2. Découpage de l'image en patches
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

# 3. Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention = F.softmax(attn_logits, dim=-1)
    return torch.matmul(attention, v)

# 4. Projection QKV
class QKVProjection(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.proj = nn.Linear(d_in, d_out * 3)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.proj(x).reshape(batch_size, seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        return qkv.chunk(3, dim=-1)

# 5. Multi-Head Attention
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.in_proj = QKVProjection(input_dim, embed_dim, num_heads)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        q, k, v = self.in_proj(x)
        attn = scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(attn)

# 6. FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 7. Transformer Block
class Transformer(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.attn = MultiheadAttention(dim, dim, heads)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# 8. TowerViT
class TowerViT(nn.Module):
    def __init__(self, dim, heads, hidden_dim, depth):
        super().__init__()
        self.layers = nn.Sequential(*[Transformer(dim, heads, hidden_dim) for _ in range(depth)])
    
    def forward(self, x):
        return self.layers(x)

# 9. Vision Transformer
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256, depth=6, heads=8, hidden_dim=512, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.transformer = nn.Sequential(*[Transformer(embed_dim, heads, hidden_dim) for _ in range(depth)])
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)

# 10. Évaluation
def evaluate(model, testloader, device, do_print=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if do_print:
        print(f'Accuracy: {100 * correct / total:.2f}%')
    return 100 * correct / total
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 11. Entraînement
# model = ViT().to(device)
# total_params = count_parameters(model)
# print(f"Total parameters: {total_params}")
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

def train(model, trainloader, testloader, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        evaluate(model, testloader)

# trainloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# testloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=False, num_workers=4)
# train(model, trainloader, testloader)

###
# Le moment où mon accuracy est au plus haut est à l'époque 19 avec une accuracy de 57.25%
# Après cette époque, l'accuracy commence à diminuer. Mais la loss continue à diminuer.
# Cela peut être dû à un overfitting du modèle.
###


# -------------------------------------------
# Main Training Loop
# -------------------------------------------
def wait(time_ms):
    """Wait for a given time in milliseconds."""
    start = time.time()
    while time.time() - start < time_ms / 1000:
        pass

def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_parameters(num_params):
    """Format the number of parameters into a human-readable string."""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"

def train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs, device):
    """Training loop."""
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, testloader, device)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

def validate(model, dataloader, device):
    """Validation loop."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Device selection
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("-----------------------------")
        print("Using NVIDIA CUDA backend for GPU computations")
        print("-----------------------------")
  
    # Check if using Linux and if ROCm is installed
    if os.name == "posix":
        try:
            if torch.backends.hip.is_available():
                device = "hip"
                print("-----------------------------")
                print("Using AMD ROCm backend for GPU computations")
                print("-----------------------------")
        except:
            pass

    if device == "cpu":
        print("-----------------------------")
        print("Using CPU")
        print("-----------------------------")
        
    wait(1000)

    # Model selection
    print("-----------------------------")
    print("Which type of model do you want to train?")
    print("-----------------------------")
    print("1. ViT (Small)")
    print("2. ViT (Medium)")
    print("3. ViT (Large)")
    print("-----------------------------")
    print("> ", end="")
    model_choice = int(input())

    # Define model hyperparameters based on choice
    if model_choice == 1:
        embed_dim = 64
        depth = 6
        heads = 8
        hidden_dim = 128
    elif model_choice == 2:
        embed_dim = 128
        depth = 12
        heads = 16
        hidden_dim = 256
    elif model_choice == 3:
        embed_dim = 256
        depth = 24
        heads = 32
        hidden_dim = 512
    else:
        raise ValueError("Invalid model choice.")

    # Create the model
    model = ViT(embed_dim=embed_dim, depth=depth, heads=heads, hidden_dim=hidden_dim).to(device)

    # Print total number of parameters
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {format_parameters(total_params)}")

    # Training settings
    print("-----------------------------")
    print("How many epochs do you want to train for?")
    print("-----------------------------")
    print("> ", end="")
    epochs = None
    while epochs is None:
        try:
            epochs = int(input())
        except ValueError:
            print("Please enter a valid integer.")

    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
    
    # Learning rate settings
    print("-----------------------------")
    print("What learning rate do you want the scheduler to start with?")
    print("-----------------------------")
    print("> ", end="")
    lr = None
    while lr is None:
        try:
            lr = float(input())
        except ValueError:
            print("Please enter a valid floating-point number.")

    # Loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epochs)

    # Training loop
    print("-----------------------------")
    print("Starting training...")
    print("-----------------------------")
    train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs, device)

    # Validation
    print("-----------------------------")
    print("Validating...")
    train_acc = validate(model, trainloader, device)
    test_acc = validate(model, testloader, device)
    print(f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()