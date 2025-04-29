import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class HemgAudioDataset(Dataset):
    def __init__(self, csv_path="data/audio_data/Hemg/labels.csv",
                audio_root="data/audio_data/Hemg/train", 
                split="train", 
                transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)
        self.audio_root = audio_root
        self.transform = transform

        # 标签映射
        self.label2emotion = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Suprised']
        self.emotion2sentiment = {
            'Happy': 'positive',
            'Suprised': 'positive',
            'Neutral': 'neutral',
            'Angry': 'negative',
            'Disgusted': 'negative',
            'Fearful': 'negative',
            'Sad': 'negative'
        }
        self.sentiment2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_root, row["file_name"])
        label = int(row["label"])

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # shape → [1, T]
            waveform = waveform.squeeze(0)
        else:
            waveform = waveform.squeeze(0)

        if self.transform:
            waveform = self.transform(waveform)


        # 情绪标签
        emotion = self.label2emotion[label]
        sentiment = self.emotion2sentiment[emotion]
        sentiment_label = self.sentiment2id[sentiment]

        return waveform, label, sentiment_label

if __name__ == "__main__":
    dataset = HemgAudioDataset()

    # 打印数据集大小
    print(f"📦 数据集大小：{len(dataset)}")
    print(dataset[0])  # 打印音频数据的形状