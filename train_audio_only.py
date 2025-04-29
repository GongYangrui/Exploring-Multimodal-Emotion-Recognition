from model import AudioEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from hemg_dataset import HemgAudioDataset
from torch.utils.data import random_split

def collate_fn(batch):
    waveforms, labels, sentiments = zip(*batch)  # 列表解包

    # Apply processor to batch (with padding)
    inputs = processor(
        [w.numpy() for w in waveforms],
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",  # or "max_length"
        max_length=80000,
        truncation=True
    )

    # last_hidden_state → 特征由模型外部提取（或提取器提前）
    return {
        "input_values": inputs.input_values,        # [B, T]
        "attention_mask": inputs.attention_mask,    # [B, T]
        "labels": torch.tensor(labels),
        "sentiments": torch.tensor(sentiments),
    }

model = AudioEncoder()
image_state_dict = torch.load("image_encoder_new.pth", map_location="cpu")
audio_state_dict = model.state_dict()

filtered_state_dict = {
    k: v for k, v in image_state_dict.items()
    if(k in audio_state_dict and 
       v.shape == audio_state_dict[k].shape 
       and '.ffn.' not in k and
       "cls_token" not in k
    )
}

load_result = model.load_state_dict(filtered_state_dict, strict=False)
# print(f"✅ 加载了 {len(filtered_state_dict)} 个共享参数")
# print("📌 未加载的参数:", load_result.missing_keys)
# print("📌 多余的参数:", load_result.unexpected_keys)

for name, param in model.named_parameters():
    if (
        'ffn' in name
        or 'classifier' in name
        or 'audio_cls_token' in name
        or 'fusion_cls_token' in name
        or 'pos_embed' in name
        or 'projection' in name
    ):
        param.requires_grad = True
        # print(f"🟢 保持可训练: {name}")
    else:
        param.requires_grad = False
        # print(f"🔒 已冻结: {name}")

# print("🔍 模型中所有参数（含可训练与冻结）:")
# for name, param in model.named_parameters():
#     print(f"{'✅ 可训练' if param.requires_grad else '❌ 冻结'} → {name:60s} | shape: {tuple(param.shape)}")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
hubert.eval().to(device)  # or .to(device)


dataset = HemgAudioDataset()
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=1e-2
)

best_val_acc = 0
patience = 5
patience_counter = 0
num_epochs = 30

model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader_train, desc=f"[Epoch {epoch+1}/{num_epochs}]")
    for batch in pbar:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["sentiments"].to(device)

        with torch.no_grad():
            hubert_outputs = hubert(input_values, attention_mask=attention_mask)
            input_values = hubert_outputs.last_hidden_state


        optimizer.zero_grad()
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=correct/total)

    train_acc = correct / total
    print(f"✅ Epoch {epoch+1}: Train Loss = {total_loss/total:.4f}, Train Acc = {train_acc:.4f}")

    # ==== Validation ====
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader_test:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["sentiments"].to(device)
            with torch.no_grad():
                hubert_outputs = hubert(input_values, attention_mask=attention_mask)
                input_values = hubert_outputs.last_hidden_state

            outputs = model(input_values)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"🔍 Validation: Loss = {val_loss / val_total:.4f}, Acc = {val_acc:.4f}")

    # ==== Early Stopping ====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_audio_model.pth")
        print("📌 当前模型为最佳模型，已保存")
    else:
        patience_counter += 1
        print(f"⚠️ 准确率未提升（{patience_counter}/{patience}）")
        if patience_counter >= patience:
            print("⏹️ 提前停止训练（Early Stopping）")
            break

print(f"🏁 训练完成，最佳验证准确率为 {best_val_acc:.4f}")


