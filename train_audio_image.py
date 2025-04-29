from model import ImageAudioEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from collections import OrderedDict
from dataset import DFEWDataset, CaerDataset
import timm

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])  # (B, 3, 224, 224)

    input_values = torch.stack([item["audio"]["input_values"] for item in batch])  # (B, 96000)
    attention_mask = torch.stack([item["audio"]["attention_mask"] for item in batch])  # (B, 96000)

    sentiments = torch.tensor([item["sentiment"] for item in batch], dtype=torch.long)

    return {
        "image": images,
        "audio": {
            "input_values": input_values,
            "attention_mask": attention_mask
        },
        "sentiment": sentiments
    }

model = ImageAudioEncoder()
mode = "21"

audio_ckpt = torch.load("expert_FFN/best_audio_model.pth", map_location="cpu")
image_ckpt = torch.load("expert_FFN/image_encoder_new.pth", map_location="cpu")
model_dict = model.state_dict()
print("📦 当前模型的所有参数（名称与形状）：\n")

shared_keys = [k for k in image_ckpt.keys() if 
               (".attn." in k or ".norm" in k)]


updated_dict = OrderedDict()

if mode == "12":
    layer_mapping = {
        "fusion_encoder.audio_expert": 0,
        "fusion_encoder.image_expert": 0,
        "fusion_encoder.fusion_layers.0": 1,
        "fusion_encoder.fusion_layers.1": 2
    }

    for target_prefix, encoder_layer_idx in layer_mapping.items():
        for attn_norm in ["attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias",
                        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            
            ckpt_key = f"encoder.layers.{encoder_layer_idx}.{attn_norm}"
            model_key = f"{target_prefix}.{attn_norm}"

            if ckpt_key in image_ckpt and model_key in model_dict:
                if image_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                    updated_dict[model_key] = image_ckpt[ckpt_key]
                    print(f"✅ {model_key} ← {ckpt_key}")

    for part in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        ckpt_key = f"encoder.layers.0.ffn.ffn.{part}"
        model_key = f"fusion_encoder.image_expert.ffn.ffn.{part}"
        if ckpt_key in image_ckpt and model_key in model_dict:
            if image_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                updated_dict[model_key] = image_ckpt[ckpt_key]
                print(f"✅ {model_key} ← {ckpt_key}")

    for part in ["0.weight", "0.bias", "2.weight", "2.bias"]:
        ckpt_key = f"encoder.layers.0.ffn.ffn.{part}"
        model_key = f"fusion_encoder.audio_expert.ffn.ffn.{part}"
        if ckpt_key in audio_ckpt and model_key in model_dict:
            if audio_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                updated_dict[model_key] = audio_ckpt[ckpt_key]
                print(f"✅ {model_key} ← {ckpt_key}")
elif mode == "21":
    layer_mapping = {
        "fusion_encoder.audio_expert.0": 0,
        "fusion_encoder.image_expert.0": 0,
        "fusion_encoder.audio_expert.1": 1,
        "fusion_encoder.image_expert.1": 1,
        "fusion_encoder.fusion_layers.1": 1,
    }
    for target_prefix, encoder_layer_idx in layer_mapping.items():
        for attn_norm in ["attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias",
                        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            
            ckpt_key = f"encoder.layers.{encoder_layer_idx}.{attn_norm}"
            model_key = f"{target_prefix}.{attn_norm}"

            if ckpt_key in image_ckpt and model_key in model_dict:
                if image_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                    updated_dict[model_key] = image_ckpt[ckpt_key]
                    print(f"✅ {model_key} ← {ckpt_key}")
    for layer_id in range(2):
        for part in ["0.weight", "0.bias", "2.weight", "2.bias"]:
            ckpt_key = f"encoder.layers.{layer_id}.ffn.ffn.{part}"
            model_key = f"fusion_encoder.audio_expert.{layer_id}.ffn.ffn.{part}"
            if ckpt_key in audio_ckpt and model_key in model_dict:
                if audio_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                    updated_dict[model_key] = audio_ckpt[ckpt_key]
                    print(f"✅ {model_key} ← {ckpt_key}")
    for layer_id in range(2):
        for part in ["0.weight", "0.bias", "2.weight", "2.bias"]:
            ckpt_key = f"encoder.layers.{layer_id}.ffn.ffn.{part}"
            model_key = f"fusion_encoder.audio_expert.{layer_id}.ffn.ffn.{part}"
            if ckpt_key in image_ckpt and model_key in model_dict:
                if image_ckpt[ckpt_key].shape == model_dict[model_key].shape:
                    updated_dict[model_key] = image_ckpt[ckpt_key]
                    print(f"✅ {model_key} ← {ckpt_key}")


special_keys = {
    "image_cls_token": "image_cls_token",
    "pos_embed": "pos_embed_image"
}

print("\n📦 加载 image_cls_token 和 pos_embed：\n")
for ckpt_key, model_key in special_keys.items():
    if ckpt_key in image_ckpt and model_key in model_dict:
        if image_ckpt[ckpt_key].shape == model_dict[model_key].shape:
            updated_dict[model_key] = image_ckpt[ckpt_key]
            print(f"✅ {model_key} ← {ckpt_key}")

special_keys = {
    "audio_cls_token": "audio_cls_token",
    "pos_embed": "pos_embed_audio",
    "fusion_cls_token": "fusion_cls_token",
    "projection.weight": "projection.weight",
    "projection.bias": "projection.bias"
}

print("\n📦 加载 audio_cls_token 和 pos_embed：\n")
for ckpt_key, model_key in special_keys.items():
    if ckpt_key in audio_ckpt and model_key in model_dict:
        if audio_ckpt[ckpt_key].shape == model_dict[model_key].shape:
            updated_dict[model_key] = audio_ckpt[ckpt_key]
            print(f"✅ {model_key} ← {ckpt_key}")


model_dict.update(updated_dict)
model.load_state_dict(model_dict, strict=False)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model.to(device)


defw_dataset = DFEWDataset()
caerdataset = CaerDataset()
combined_training_dataset = ConcatDataset([defw_dataset, caerdataset])

test_dataset = CaerDataset(split="test")

train_loader = DataLoader(
    combined_training_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)
test_loader = DataLoader(test_dataset, 
                         batch_size=32, 
                         shuffle=False, 
                         num_workers=4, 
                         collate_fn=collate_fn, 
                         pin_memory=True)

image_embedding_model = timm.create_model('vit_base_patch16_224', pretrained=True) # 会自动在embedding序列前中添加一个cls_token
image_embedding_model.eval()
image_embedding_model.to(device)

audio_embedding_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft") # 加载 HuBERT 模型
audio_embedding_model.eval()
audio_embedding_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

num_epochs = 30  # 你可以改
early_stop_patience = 10
best_test_loss = float('inf')
no_improve_count = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device)  # (B, 3, 224, 224)
        input_values = batch["audio"]["input_values"].to(device)  # (B, 96000)
        attention_mask = batch["audio"]["attention_mask"].to(device)
        labels = batch["sentiment"].to(device)  # (B,)

        with torch.no_grad():
            images = image_embedding_model.patch_embed(images) 

        # === 2. 音频特征 ===
        with torch.no_grad():
            audio_output = audio_embedding_model(input_values=input_values, attention_mask=attention_mask)
            audio_feat = audio_output.last_hidden_state  # (B, T, 1024)

        # === 3. 前向传播 ===
        outputs = model(audio_feat, images)  # (B, 3)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # === 4. 反向传播 ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = correct / total
        pbar.set_postfix(loss=total_loss/total, acc=acc)


    print(f"Epoch {epoch+1}: Loss = {total_loss/total:.4f}, Acc = {correct/total:.4f}")

    model.eval()
    val_total = 0
    val_correct = 0
    val_total_loss = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[Eval] Epoch {epoch+1}"):
            images = batch["image"].to(device)
            input_values = batch["audio"]["input_values"].to(device)
            attention_mask = batch["audio"]["attention_mask"].to(device)
            labels = batch["sentiment"].to(device)

            images = image_embedding_model.patch_embed(images) 
            audio_output = audio_embedding_model(input_values=input_values, attention_mask=attention_mask)
            audio_feat = audio_output.last_hidden_state

            outputs = model(audio_feat, images)
            loss = criterion(outputs, labels)
            val_total_loss += loss.item() * labels.size(0)
            val_preds = outputs.argmax(dim=1)
            val_correct += (val_preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_total_loss / val_total
    print(f"Validation: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")

    if val_loss < best_test_loss:
        best_test_loss = val_loss
        no_improve_count = 0
        print("✅ Test loss improved, saving model...")
        torch.save(model.state_dict(), "image_audio_encoder_2_1_caer_dfew.pth")
    else:
        no_improve_count += 1
        print(f"⏸ No improvement. Patience: {no_improve_count}/{early_stop_patience}")
        if no_improve_count >= early_stop_patience:
            print("🛑 Early stopping triggered!")
            break