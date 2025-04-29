import torch.nn as nn
import random
import torch

class LlavaAdapter(nn.Module):
    def __init__(self, llava_model, llava_tokenizer,fused_dim, hidden_dim, num_classes=4, dropout=0.2):
        super().__init__()
        self.llava_model = llava_model  # 预加载好的LLaVA
        self.llava_tokenizer = llava_tokenizer  # 预加载好的LLaVA tokenizer
        self.fused_projector = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.GELU(),  # 比ReLU更滑顺，Transformer常用GELU
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim // 2),  # 也可以加一层归一化，稳定训练

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim)  # 也可以加一层归一化，稳定训练
        )  # 把cls token调整到LLaVA需要的大小
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        ) 
        # self.classifier = self.classifier.half()

    def forward(self, sentences, fused_features):
        """
        sentences: list of str
        projected_fused: tensor [B, fused_dim]
        """
        prompt_templates = [
            "Based on the multimodal inputs, classify the emotion.",
            "From the provided information, select the correct emotion: ",
            "Analyze the following input and classify its emotion: ",
            "Based on the provided sentence, audio, and image features, classify the emotion expressed: ",
            "Given the multimodal information (text, audio, visual), predict the speaker's emotional state: ",
            "Analyze the sentence along with corresponding audio and visual cues to determine the emotion: ",
            "Classify the emotional state by integrating the text, audio signals, and visual features: ",
            "Using the combined features from audio, image, and text, identify the expressed emotion: ",
            "Predict the emotion based on the multimodal (sentence + audio + image) input: ",
            "From the multimodal context provided (text, audio, visual), select the correct emotion category: ",
            "Determine the emotional expression considering both linguistic and non-linguistic cues: ",
            "Given the sentence and its associated audio-visual features, classify the emotion appropriately: ",
            "Taking into account the multimodal inputs, infer the speaker's emotional state: ",
        ]
        prompted_sentences = [random.choice(prompt_templates) +s + "From the following labels, select one emotion to answer: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']" for s in sentences]
        batch_encoding = self.llava_tokenizer(
            prompted_sentences,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=512    # 防止爆显存，你可以按需要调
        )
        # fused_features传进来的时候在cuda:2上
        input_ids = batch_encoding["input_ids"].to(fused_features.device)
        attention_mask = batch_encoding["attention_mask"].to(fused_features.device)
        with torch.no_grad():
            input_embeds = self.llava_model.language_model.model.embed_tokens(input_ids) # 执行完之后在cuda:0上
        
        # fused_features torch.Size([16, 768])
        # projected_fused torch.Size([16, 4096])
        projected_fused = self.fused_projector(fused_features).to(input_embeds.device) # 执行完之后在cuda:0上
        # projected_fused = projected_fused.unsqueeze(1).half() 
        projected_fused = projected_fused.unsqueeze(1)
        if torch.isnan(projected_fused).any() or torch.isinf(projected_fused).any():
            print("!!! Found NaN or Inf in fused_features !!!")

        fused_inputs_embeds = torch.cat([projected_fused, input_embeds], dim=1)
        # attention_mask torch.Size([16, 95])
        # multimodal_mask: torch.Size([16, 1])
        multimodal_mask = torch.ones(projected_fused.size(0), projected_fused.size(1), device=attention_mask.device)
        new_attention_mask = torch.cat([multimodal_mask, attention_mask], dim=1)
        del input_embeds
        del projected_fused
        torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = self.llava_model(
                inputs_embeds=fused_inputs_embeds,
                attention_mask=new_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1] # ([16, 122, 4096]) 在设备cuda0上
        fused_hidden = hidden_states[:, 0, :] # ([16, 4096])
        fused_hidden = fused_hidden.to(self.classifier[-1].weight.device)
        logits = self.classifier(fused_hidden)
        return logits
