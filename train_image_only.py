from model import ImageEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from affect_dataset import AffectNetDataset
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm
import timm


affectnet = AffectNetDataset(
    csv_path="data/facial_image/affectnet/labels.csv",
    image_root="data/facial_image/affectnet",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
train_ratio = 0.8
train_size = int(train_ratio * len(affectnet))
test_size = len(affectnet) - train_size
train_dataset, test_dataset = random_split(affectnet, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model = ImageEncoder()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

image_embedding_model = timm.create_model('vit_base_patch16_224', pretrained=True) # 会自动在embedding序列前中添加一个cls_token
image_embedding_model.eval()
image_embedding_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

best_acc = 0
patience = 10
patience_counter = 0
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")
    for images, _, sentiments in pbar:
        images = images.to(device)
        sentiments = sentiments.to(device)
        with torch.no_grad():
            images = image_embedding_model.patch_embed(images) 
    
        optimizer.zero_grad()
        outputs = model(images)  # logits: [B, 3]
        loss = criterion(outputs, sentiments)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sentiments.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == sentiments).sum().item()
        total += sentiments.size(0)

        acc = correct / total
        pbar.set_postfix(loss=total_loss/total, acc=acc)

    print(f"✅ Epoch {epoch+1}: Loss = {total_loss/total:.4f}, Acc = {acc:.4f}")

    model.eval()
    val_total = 0
    val_correct = 0
    val_total_loss = 0

    with torch.no_grad():
        for images, _, sentiments in test_loader:
            images = images.to(device)
            sentiments = sentiments.to(device)
            with torch.no_grad():
                images = image_embedding_model.patch_embed(images) 

            outputs = model(images)
            loss = criterion(outputs, sentiments)

            val_total_loss += loss.item()
            val_preds = outputs.argmax(dim=1)
            val_correct += (val_preds == sentiments).sum().item()
            val_total += sentiments.size(0)

    val_acc = val_correct / val_total
    val_loss = val_total_loss / len(test_loader)
    print(f"🔍 Validation: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")

    # ===== Early Stopping 判断 =====
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "image_encoder_new.pth")
        print("📌 当前模型为最佳模型，已保存")
    else:
        patience_counter += 1
        print(f"⚠️ 准确率未提升（{patience_counter}/{patience}）")

    if patience_counter >= patience:
        print("🛑 提前终止训练：验证集准确率未提升")
        break
