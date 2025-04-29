from dataset import ImageAndAudioDataset
from model import ImageAudioEncoder
import torch
import timm
from transformers import HubertModel
from utils import load_llava_model, evaluate
from llava_adapter import LlavaAdapter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

base_path="test/MELD-RAW"
meld_train_dataset = ImageAndAudioDataset(
    base_path=base_path,
    split="train",
    image_transform=None,
    audio_transform=None
)

meld_test_dataset = ImageAndAudioDataset(base_path, 
                                        split="test", 
                                        image_transform=None, 
                                        audio_transform=None)

train_loader = DataLoader(
    meld_train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4
)
test_loader = DataLoader(
    meld_test_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4
)


device1 = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

audio_image_encoder = ImageAudioEncoder()
audio_image_encoder_load_path = "expert_FFN/image_audio_encoder_2_1_caer_dfew.pth"
audio_image_encoder.load_state_dict(torch.load(audio_image_encoder_load_path))
audio_image_encoder.eval()
audio_image_encoder.to(device1)

image_embedding_model = timm.create_model('vit_base_patch16_224', pretrained=True) # 会自动在embedding序列前中添加一个cls_token
image_embedding_model.eval()
image_embedding_model.to(device1)

audio_embedding_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft") # 加载 HuBERT 模型
audio_embedding_model.eval()
audio_embedding_model.to(device1)


llava_model, llava_tokenizer = load_llava_model(model_name="llava-hf/llava-1.5-7b-hf")

# 冻结 LLaVA 模型的参数
for param in llava_model.parameters():
    param.requires_grad = False

llava_adapter = LlavaAdapter(
    llava_model=llava_model,
    llava_tokenizer=llava_tokenizer,
    fused_dim=768,        # 你ImageAudioEncoder的输出维度
    hidden_dim=4096      # LLaVA视觉输入要求768（比如ViT base）
)
llava_adapter.fused_projector = llava_adapter.fused_projector.to(device2)
llava_adapter.classifier = llava_adapter.classifier.to(device2)


optimizer = torch.optim.AdamW([
    {'params': audio_image_encoder.parameters(), 'lr': 2e-5},
    {'params': llava_adapter.fused_projector.parameters(), 'lr': 2e-4},
    {'params': llava_adapter.classifier.parameters(), 'lr': 2e-4}
], weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

best_val_loss = float('inf')
patience = 5  # 连续多少个epoch val_loss不下降就停
patience_counter = 0

num_epochs = 30

wandb.init(
    project="multimodal-emotion-recognition",
    name=f"llava_adapter_fine_tune_on_4_emotions",
    config={
        "learning_rates": {
            "audio_image_encoder": 2e-5,
            "fused_projector": 2e-4,
            "classifier": 2e-4
        },
        "epochs": num_epochs,
        "batch_size": 32,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 0.01,
    }
)

for epoch in range(num_epochs):
    audio_image_encoder.train()
    llava_adapter.train()

    running_loss = 0.0
    running_loss_per_batch = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]")
    for batch in pbar:
        images = batch["image"].to(device1)
        audio_input_values = batch["audio"]["input_values"].to(device1)
        attention_mask = batch["audio"]["attention_mask"].to(device1)
        sentences = batch["sentence"]  # list of str
        emotion_labels = batch["emotion"].to(device2)

        with torch.no_grad():
            images = image_embedding_model.patch_embed(images) 

        with torch.no_grad():
            audio_output = audio_embedding_model(input_values=audio_input_values, attention_mask=attention_mask)
            audio_feat = audio_output.last_hidden_state  # (B, T, 1024)
    

        optimizer.zero_grad()

        # (1) ImageAudioEncoder提取cls token
        fused_feature = audio_image_encoder(audio_feat, images, return_fuesd=True)  # [B, 768]
        fused_feature = fused_feature.to(device2)  # 转移到LLaVA所在的GPU
        # (2) LlavaAdapter进行推理
        outputs = llava_adapter(sentences, fused_feature)

        logits = outputs  # [B, 7]

        # (3) Cross Entropy Loss
        loss = F.cross_entropy(logits, emotion_labels)

        # (4) 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_per_batch += loss.item() * emotion_labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == emotion_labels).sum().item()
        total += emotion_labels.size(0)
        pbar.set_postfix(loss=running_loss_per_batch/total, acc=correct/total)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    wandb.log({
        "train_loss": avg_loss,
        "train_accuracy": acc * 100
    })
    print(f"Training: epoch {epoch+1} - Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    audio_image_encoder.eval()
    llava_adapter.eval()
    torch.cuda.empty_cache()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device1)
            audio_inputs = batch["audio"]["input_values"].to(device1)
            attention_mask = batch["audio"]["attention_mask"].to(device1)
            sentences = batch["sentence"]
            emotion_labels = batch["emotion"].to(device2)

            with torch.no_grad():
                images = image_embedding_model.patch_embed(images) 

            with torch.no_grad():
                audio_output = audio_embedding_model(input_values=audio_inputs, attention_mask=attention_mask)
                audio_feat = audio_output.last_hidden_state

            fused_feature = audio_image_encoder(audio_feat, images, return_fuesd=True)
            fused_feature = fused_feature.to(device2)

            outputs = llava_adapter(sentences, fused_feature)
            logits = outputs

            loss = F.cross_entropy(logits, emotion_labels)
            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == emotion_labels).sum().item()
            total += emotion_labels.size(0)

    avg_loss = running_loss / len(test_loader)
    acc = correct / total
    scheduler.step(avg_loss)
    wandb.log({
        "test_loss": avg_loss,
        "test_accuracy": acc * 100
    })

    print(f"Validation: epoch {epoch+1} - Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
    
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        patience_counter = 0

        # 保存当前最好模型
        save_dir = "expert_FFN/save_dir"
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "audio_image_encoder": audio_image_encoder.state_dict(),
            "llava_adapter": llava_adapter.state_dict(),
        }, os.path.join(save_dir, f"best_model_4.pth"))
        print("✅ Best model updated.")
    else:
        patience_counter += 1
        print(f"⚠️ Early stopping patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        wandb.finish()
        print("⛔ Early stopping triggered!")
        break








