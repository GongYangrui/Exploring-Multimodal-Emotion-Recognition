import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        return self.ffn(x)
    
class FusionFFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x_fused): # x_fused = x_img + x_aud (B, N + 1 + M + 2, D)
        return self.ffn(x_fused)
    
class CustomExpertLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, drop_out=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=drop_out)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = ExpertFFN(hidden_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + self.dropout(attn_out)

        x_norm2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm2))
        return x
    

class CustomFusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fusion_ffn = FusionFFN(hidden_dim)

    def forward(self, x_fused):
        x_norm1 = self.norm1(x_fused)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x_fused = x_fused + self.dropout1(attn_out)

        x_norm2 = self.norm2(x_fused)
        x_fused = x_fused + self.dropout2(self.fusion_ffn(x_norm2))
        return x_fused

class SingleModalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomExpertLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # self.audio_expert = CustomExpertLayer(hidden_dim, num_heads)
        # self.image_expert = CustomExpertLayer(hidden_dim, num_heads)

        # self.fusion_layers = nn.ModuleList([
        #     CustomFusionLayer(hidden_dim, num_heads),
        #     CustomFusionLayer(hidden_dim, num_heads)
        # ])
        self.audio_expert = nn.Sequential(
            CustomExpertLayer(hidden_dim, num_heads),
            CustomExpertLayer(hidden_dim, num_heads)
        )

        self.image_expert = nn.Sequential(
            CustomExpertLayer(hidden_dim, num_heads),
            CustomExpertLayer(hidden_dim, num_heads)
        )

        self.fusion_layers = nn.ModuleList([
            CustomFusionLayer(hidden_dim, num_heads)
        ])

    def forward(self, x_aud, x_img):
        x_img = self.audio_expert(x_img)
        x_aud = self.image_expert(x_aud)

        # 拼接：沿 sequence 维度（dim=1）
        x_fused = torch.cat([x_aud, x_img], dim=1)

        # 再通过两层 Fusion Layer
        for layer in self.fusion_layers:
            x_fused = layer(x_fused)

        return x_fused

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=3, num_classes=3, num_patches=196, dropout_prob=0.1):
        super().__init__()
        self.encoder = SingleModalEncoder(hidden_dim, num_heads)
        self.image_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在图片最前面
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_img):
        B = x_img.shape[0]
        cls = self.image_cls_token.expand(B, -1, -1)  # [B, 1, D]
        x_img = torch.cat((cls, x_img), dim=1)
        x_img = x_img + self.pos_embed[:, :x_img.size(1), :]
        x_img = self.encoder(x_img)
        cls = x_img[:, 0]  # [I_CLS]
        cls = self.dropout(cls)
        return self.classifier(cls)
    
class AudioEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=3, num_classes=3, dropout_prob=0.1, max_len=256):
        super().__init__()
        self.input_dim = 1024
        self.projection = nn.Linear(self.input_dim, hidden_dim)
        self.encoder = SingleModalEncoder(hidden_dim, num_heads)
        self.audio_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频最前面
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频和图片中间
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim)) 
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_aud):
        B = x_aud.shape[0]
        x_aud = self.projection(x_aud)
        cls = self.audio_cls_token.expand(B, -1, -1)
        seq = self.fusion_cls_token.expand(B, -1, -1)
        x_aud = torch.cat((cls, x_aud, seq), dim=1)
        x_aud = x_aud + self.pos_embed[:, :x_aud.size(1), :]
        x_aud = self.encoder(x_aud)
        cls = x_aud[:, 0]  # [T_CLS]
        cls = self.dropout(cls)
        return self.classifier(cls)
    
class ImageAudioEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=3, num_classes=3, dropout_prob=0.1, max_len=256, num_patches=196):
        super().__init__()
        self.fusion_encoder = FusionEncoder(hidden_dim, num_heads)
        self.aud_input_dim = 1024
        self.projection = nn.Linear(self.aud_input_dim, hidden_dim)
        self.image_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在图片最前面
        self.audio_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频最前面
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频和图片中间
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

        self.pos_embed_audio = nn.Parameter(torch.randn(1, max_len, hidden_dim)) 
        self.pos_embed_image = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))

    def forward(self, x_aud, x_img, return_fuesd=False):
        B = x_img.shape[0]
        N = x_img.shape[1]
        M = x_aud.shape[1]
        x_aud = self.projection(x_aud)
        cls_img = self.image_cls_token.expand(B, -1, -1)
        cls_aud = self.audio_cls_token.expand(B, -1, -1)
        cls_fusion = self.fusion_cls_token.expand(B, -1, -1)
        x_img = torch.cat((cls_img, x_img), dim=1)
        x_aud = torch.cat((cls_aud, x_aud), dim=1)
        x_aud = torch.cat((x_aud, cls_fusion), dim=1)
        x_img = x_img + self.pos_embed_image[:, :x_img.size(1), :]
        x_aud = x_aud + self.pos_embed_audio[:, :x_aud.size(1), :]
        x_fuesd = self.fusion_encoder(x_aud, x_img) # [B, M + 2 + N + 1, D]
        cls = x_fuesd[:, M + 1]
        cls = self.dropout(cls)
        if return_fuesd:
            return cls
        else:
            return self.classifier(cls)





class ModalityRouter(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=4, num_classes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_encoder = SingleModalEncoder(hidden_dim, num_heads)
        self.audio_encoder = SingleModalEncoder(hidden_dim, num_heads)
        self.fusion_layer = FusionEncoder(hidden_dim, num_heads)

        self.image_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在图片最前面
        self.audio_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频最前面
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) # 放在音频和图片中间

        self.classifier = nn.Linear(hidden_dim, num_classes)



    def forward(self, x_img=None, x_aud=None, mode='fusion'):
        if mode == 'image_only':
            B = x_img.shape[0]
            cls = self.image_cls_token.expand(B, -1, -1)  # [B, 1, D]
            x_img = torch.cat((cls, x_img), dim=1)
            x_img = self.image_encoder(x_img)
            cls = x_img[:, 0]  # [I_CLS]
            # return self.image_encoder(x_img)
        elif mode == 'audio_only':
            B = x_aud.shape[0]
            cls = self.audio_cls_token.expand(B, -1, -1)
            x_aud = torch.cat((cls, x_aud), dim=1)
            x_aud = self.audio_encoder(x_aud)
            cls = x_aud[:, 0]  # [T_CLS]
            # return self.audio_encoder(x_aud)
        elif mode == 'fusion':
            B = x_img.shape[0]
            N = x_img.shape[1]
            M = x_aud.shape[1]
            cls_img = self.image_cls_token.expand(B, -1, -1)
            cls_aud = self.audio_cls_token.expand(B, -1, -1)
            cls_fusion = self.fusion_cls_token.expand(B, -1, -1)
            x_img = torch.cat((cls_img, x_img), dim=1)
            x_aud = torch.cat((cls_aud, x_aud), dim=1)
            x_aud = torch.cat((x_aud, cls_fusion), dim=1)
            x_fuesd = self.fusion_layer(x_aud, x_img) # [B, M + 2 + N + 1, D]
            cls = x_fuesd[:, M + 1]
            # return self.fusion_layer(x_img_feat, x_aud_feat)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.classifier(cls)



if __name__ ==  "__main__":
    model = ModalityRouter()

    x_img = torch.randn(4, 16, 768)  # 4 batch, 16 tokens
    x_aud = torch.randn(4, 16, 768)

    # image only
    out_img = model(x_img=x_img, mode='image_only')
    print(out_img.shape)

    # audio only
    out_aud = model(x_aud=x_aud, mode='audio_only')
    print(out_aud.shape)

    # fusion
    out_fused = model(x_img=x_img, x_aud=x_aud, mode='fusion')
    print(out_fused.shape)