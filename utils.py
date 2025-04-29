from transformers import LlavaForConditionalGeneration, AutoTokenizer
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F


def load_llava_model(model_name="llava-hf/llava-1.5-7b-hf", device="cuda"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )

    return model, tokenizer

@torch.no_grad()
def evaluate(audio_image_encoder, llava_adapter, val_loader, device1, device2):
    audio_image_encoder.eval()
    llava_adapter.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device1)
        audio_inputs = batch["audio"]["input_values"].to(device1)
        attention_mask = batch["audio"]["attention_mask"].to(device1)
        sentences = batch["sentence"]
        emotion_labels = batch["emotion"].to(device2)

        fused_feature = audio_image_encoder(audio_inputs, images, return_fuesd=True)
        fused_feature = fused_feature.to(device2)

        outputs = llava_adapter(sentences, fused_feature)
        logits = outputs

        loss = F.cross_entropy(logits, emotion_labels)
        running_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == emotion_labels).sum().item()
        total += emotion_labels.size(0)

    avg_loss = running_loss / len(val_loader)
    acc = correct / total

    return avg_loss, acc
