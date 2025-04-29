from dataset import CAERValidationDataset
from model import ImageAudioEncoder
import torch
import timm
from transformers import HubertModel
from utils import load_llava_model, evaluate
from llava_adapter import LlavaAdapter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

device1 = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device3 = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

val_dataset = CAERValidationDataset(device=device3)
val_loader =  DataLoader(val_dataset, batch_size=1, shuffle=False)


image_embedding_model = timm.create_model('vit_base_patch16_224', pretrained=True) # 会自动在embedding序列前中添加一个cls_token
image_embedding_model.eval()
image_embedding_model.to(device1)

audio_embedding_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft") # 加载 HuBERT 模型
audio_embedding_model.eval()
audio_embedding_model.to(device1)

audio_image_encoder = ImageAudioEncoder()
llava_model, llava_tokenizer = load_llava_model(model_name="llava-hf/llava-1.5-7b-hf")

checkpoint_path = "expert_FFN/save_dir/best_model_4.pth"
llava_adapter = LlavaAdapter(
    llava_model=llava_model,
    llava_tokenizer=llava_tokenizer,
    fused_dim=768,        # 你ImageAudioEncoder的输出维度
    hidden_dim=4096      # LLaVA视觉输入要求768（比如ViT base）
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
audio_image_encoder.load_state_dict(checkpoint["audio_image_encoder"])
llava_adapter.load_state_dict(checkpoint["llava_adapter"])

llava_adapter.fused_projector = llava_adapter.fused_projector.to(device2)
llava_adapter.classifier = llava_adapter.classifier.to(device2)
audio_image_encoder = audio_image_encoder.to(device1)

audio_image_encoder.eval()
llava_adapter.eval()


all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validating"):
        images = batch['image'].to(device1)
        audio_input_values = batch["audio"]["input_values"].to(device1)
        attention_mask = batch["audio"]["attention_mask"].to(device1)
        texts = batch['text']
        labels = batch['label'].to(device2)

        with torch.no_grad():
            images = image_embedding_model.patch_embed(images) 

        with torch.no_grad():
            audio_output = audio_embedding_model(input_values=audio_input_values, attention_mask=attention_mask)
            audio_feat = audio_output.last_hidden_state  # (B, T, 1024)

        # 1. 图像特征
        image_features = images

        # 2. 音频特征
        audio_features = audio_feat

        # 3. 送到AudioImageEncoder，得到融合特征
        fused_features = audio_image_encoder(audio_features, image_features, return_fuesd=True)
        fused_features = fused_features.to(device2)
        print(f"fused_features mean: {fused_features.mean()}, std: {fused_features.std()}")


        # 4. 送到LlavaAdapter分类
        logits = llava_adapter(texts, fused_features)

        # 5. 取预测结果
        preds = torch.argmax(logits, dim=1)  # (B, )

        all_predictions.append(preds.cpu())
        all_labels.append(labels.cpu())

# 拼接所有batch
all_predictions = torch.cat(all_predictions, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# 计算准确率
accuracy = (all_predictions == all_labels).float().mean().item()

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 记录每个样本的预测
for idx in range(len(all_predictions)):
    print(f"Sample {idx}: Prediction = {all_predictions[idx].item()}, Ground Truth = {all_labels[idx].item()}")
