import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
import torchaudio
from transformers import Wav2Vec2Processor
from facenet_pytorch import MTCNN
import random
from pathlib import Path

class ImageAndAudioDataset(Dataset):
    def __init__(self, base_path, split="train", image_transform=None, audio_transform=None):
        '''
        -base_path
            -audio
                -dev
                -test
                -train 
            -image
                -dev
                -test
                -train
            -(dev/test/train)_sent_emo.csv
        '''
        if image_transform is None and split == "train":
            image_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomRotation(degrees=10),  # 轻微旋转 ±10°
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        elif image_transform is None and split != "train":
            image_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

        self.data_dir = base_path # 这里是数据集的根目录
        self.split = split
        self.image_transform = image_transform
        self.audio_transform = audio_transform

        self.image_dir = os.path.join(base_path, "face_image", split)
        self.audio_dir = os.path.join(base_path, "audio", split)
        self.csv_path = os.path.join(base_path, f"{split}_sent_emo.csv")

        csv_data = pd.read_csv(self.csv_path)
        self.data = []
        keep_emotions = {'anger', 'joy', 'neutral', 'sadness'}
        for _, row in csv_data.iterrows():
            utt_id = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            img_path = os.path.join(self.image_dir, f"{utt_id}.jpg")
            audio_path = os.path.join(self.audio_dir, f"{utt_id}.wav")
            # TODO：将标签缩减到四个
            emotion = row['Emotion']
            if emotion in keep_emotions and os.path.exists(img_path) and os.path.exists(audio_path):
                self.data.append((utt_id, row))  # 保存 ID 和 row

        # TODO: 缩减情感标签
        self.keep_emotions = sorted(['anger', 'joy', 'neutral','sadness'])
        # self.emotions = sorted(csv_data["Emotion"].dropna().unique())
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(self.keep_emotions)}
        self.idx2emotion = {idx: emotion for emotion, idx in self.emotion2idx.items()}

        self.sentiments = sorted(csv_data["Sentiment"].dropna().unique())
        self.sentiment2idx = {sent: idx for idx, sent in enumerate(self.sentiments)}
        self.idx2sentiment = {idx: sent for sent, idx in self.sentiment2idx.items()}

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft") # 语音预处理工具


    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        """
        utt_id, row = self.data[idx]

        img_path = os.path.join(self.image_dir, f"{utt_id}.jpg")
        audio_path = os.path.join(self.audio_dir, f"{utt_id}.wav")

        image = Image.open(img_path).convert("RGB")
        
        if self.image_transform:
            image = self.image_transform(image)

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # shape → (1, N) 多声道到单声道
            waveform = waveform.squeeze(0)  # shape → (N, ) 去掉第一维
        
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        processed_audio = self.processor(
                        waveform,                      # 1D tensor
                        sampling_rate=16000,
                        return_tensors="pt",
                        max_length=80000,
                        padding="max_length",
                        truncation=True
                    )
        
        emotion_str = row["Emotion"]
        sentiment_str = row["Sentiment"]

        emotion_label = torch.tensor(self.emotion2idx.get(emotion_str, -1), dtype=torch.long)
        sentiment_label = torch.tensor(self.sentiment2idx.get(sentiment_str, -1), dtype=torch.long)

        sentence = row["Utterance"].strip()  # 去除首尾空格
        
        return {
        "image": image,
        "audio": {
            "input_values": processed_audio["input_values"].squeeze(0),
            "attention_mask": processed_audio["attention_mask"].squeeze(0)
        },   
        "sentence": sentence,            # str
        "emotion": emotion_label,
        "sentiment": sentiment_label,
        "utt_id": utt_id  # 可选，用于 debug 或后续保存
        }


class DFEWDataset(Dataset):
    def __init__(self, 
                 image_root="/data/clip_224x224", 
                 audio_root="/data/DFEW-part2/Clip/original/audio", 
                 transform=None, 
                 audio_transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.image_root = image_root
        self.audio_root = audio_root
        self.transform = transform
        self.audio_transform = audio_transform
        self.csv_path = "data/DFEW-part2/Annotation/processed_annotation.csv"
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft") 

        self.sentiments = ['negative', 'neutral', 'positive']
        self.sentiment2idx = {s: i for i, s in enumerate(self.sentiments)}
        self.idx2sentiment = {i: s for s, i in self.sentiment2idx.items()}

        # 处理csv文件
        csv_data = pd.read_csv(self.csv_path)

        valid_rows = []
        for _, row in csv_data.iterrows():
            video_id = row["order"]
            frame_dir = os.path.join(self.image_root, str(video_id))
            if os.path.isdir(frame_dir) and len(os.listdir(frame_dir)) > 0:
                valid_rows.append(row)

        # 转为 DataFrame 存储
        self.csv_data = pd.DataFrame(valid_rows).reset_index(drop=True)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        video_id = str(row['order']).zfill(5)
        emotion_label = torch.tensor(row['emotion_new'], dtype=torch.long)
        sentiment_str = row['sentiment']
        sentiment_label = torch.tensor(self.sentiment2idx[sentiment_str], dtype=torch.long)

        # 图像帧路径
        frame_dir = os.path.join(self.image_root, video_id)
        image_files = [f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        selected_image = random.choice(image_files)
        image_path = os.path.join(frame_dir, selected_image)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 加载音频
        audio_id = str(int(video_id))  # 去掉前导0
        audio_path = os.path.join(self.audio_root, f"{audio_id}.wav")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # shape → (1, N) 多声道到单声道
            waveform = waveform.squeeze(0)  # shape → (N, ) 去掉第一维
        else:
            waveform = waveform.squeeze(0)
        
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        processed_audio = self.processor(
                waveform,                    # 1D tensor 一定要注意这里进来的只能是1D tensor，不然直接炸掉内存
                sampling_rate=16000,
                return_tensors="pt",
                max_length=80000,
                padding="max_length",
                truncation=True
            )

        return {
            'video_id': video_id,
            'image': image,
            "audio": {
                "input_values": processed_audio["input_values"].squeeze(0),
                "attention_mask": processed_audio["attention_mask"].squeeze(0)
            }, 
            "emotion_label": emotion_label,
            "sentiment": sentiment_label
        }
    
class CaerDataset(Dataset):
    def __init__(self, csv_path="/data/audio_image_data/caers/caer/caer_dataset.csv", 
                 base_path="/data/audio_image_data/caers/caer", 
                 transform=None, 
                 audio_transform=None,
                 split="train"):
        if transform is None and split == "train":
            transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomRotation(degrees=10),  # 轻微旋转 ±10°
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        elif transform is None and split != "train":
            transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        self.csv_path = csv_path
        self.base_path = Path(base_path)
        self.transform = transform
        self.audio_transform = audio_transform
        self.data = pd.read_csv(self.csv_path)

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft") 
        self.emotion2idx = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        self.sentiment2idx = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 拼接完整路径
        face_path = self.base_path / "caer_faces" / row['face_path']
        audio_path = self.base_path / "caer_audio" / row['audio_path']
        # 读取图片
        image = Image.open(face_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 读取音频
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # shape → (1, N) 多声道到单声道
            waveform = waveform.squeeze(0)  # shape → (N, ) 去掉第一维
        else:
            waveform = waveform.squeeze(0)
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        # 使用 HuBERT processor
        processed_audio = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            max_length=80000,
            padding="max_length",
            truncation=True
        )

        # emotion label 和 sentiment label 编码成数字
        emotion_label = self.emotion2idx[row['label']]
        sentiment_label = self.sentiment2idx[row['sentiment']]

        return {
            'image': image,
            'audio': {
                'input_values': processed_audio["input_values"].squeeze(0),
                'attention_mask': processed_audio["attention_mask"].squeeze(0)
            },
            'emotion_label': emotion_label,
            'sentiment': sentiment_label
        }


class CAERValidationDataset(Dataset):
    def __init__(self, 
        csv_path="/data/audio_image_data/caer_final_validation_dataset/caer_final_validation_metadata.csv", 
        image_transform=None, 
        audio_transform=None,
        device=None
        ):
        """
        Args:
            csv_path (str): Path to the CSV file
            image_transform (callable, optional): Transform to be applied on the image
            audio_transform (callable, optional): Transform to be applied on the audio waveform
        """
        if image_transform is None:
            image_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.data = pd.read_csv(csv_path)
        self.image_transform = image_transform
        self.audio_transform = audio_transform

        self.label_mapping = {
            'Anger': 'anger',
            'Happy': 'joy',
            'Neutral': 'neutral',
            'Sad': 'sadness'
        }

        self.target_labels = ['anger', 'joy', 'neutral', 'sadness']
        self.label2id = {label: idx for idx, label in enumerate(self.target_labels)}
        self.data = self.data[self.data['label'].isin(self.label_mapping.keys())].reset_index(drop=True)
        self.data['label'] = self.data['label'].map(self.label_mapping)  # 统一改成小写新版标签
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device=device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft") 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 读取图片
        image = Image.open(row['image_path']).convert('RGB')
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            best_idx = probs.argmax()
            best_box = boxes[best_idx]  # (x1, y1, x2, y2)

            best_box = [int(b) for b in best_box]
            x1, y1, x2, y2 = best_box
            image = image.crop((x1, y1, x2, y2))
        else:
            print("无人脸")
            image = image
        
        if self.image_transform:
            image = self.image_transform(image)

        # 读取音频
        waveform, sample_rate = torchaudio.load(row['audio_path'])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # shape → (1, N) 多声道到单声道
            waveform = waveform.squeeze(0)  # shape → (N, ) 去掉第一维
        else:
            waveform = waveform.squeeze(0)
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        processed_audio = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            max_length=80000,
            padding="max_length",
            truncation=True
        )

        # 文本
        text = row['text']

        # 标签：统一小写后，映射成数字
        label_str = row['label']
        label_id = self.label2id[label_str]
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        sample = {
            'image': image,
            'audio': {
                'input_values': processed_audio["input_values"].squeeze(0),
                'attention_mask': processed_audio["attention_mask"].squeeze(0)
            },
            'text': text,
            'label': label_tensor  # 返回int标签
        }
        
        return sample


if __name__ == "__main__":

    dataset = CAERValidationDataset(device="cuda:0")
    print(dataset[0]["audio"]["input_values"].shape)  # torch.Size([1, 80000])
    assert False



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
    defw_dataset = DFEWDataset()
    base_path="/test/MELD-RAW"
    meld_dataset = ImageAndAudioDataset(
                        base_path=base_path,
                        split="train",
                        image_transform=None,
                        audio_transform=None
                    )
    caerdataset = CaerDataset()
    combined_dataset = ConcatDataset([defw_dataset, caerdataset])
    dataloader = DataLoader(
                combined_dataset,
                batch_size=8,
                shuffle=True,
                collate_fn=collate_fn
    )


    for batch in dataloader:
        print("image:", batch["image"].shape)
        print("input_values:", batch["audio"]["input_values"].shape)
        print("attention_mask:", batch["audio"]["attention_mask"].shape)
        print("sentiment:", batch["sentiment"])
        break  # 只看一个 batch
    
