import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class AffectNetDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        """
        Args:
            csv_path (str): CSV 文件路径（包含 pth,label）
            image_root (str): 图像根目录（affectnet 文件夹）
            transform (callable, optional): 图像预处理，如 transforms.Compose
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

        # 建立 label 到 id 的映射，例如：{'anger': 0, 'fear': 1, ...}
        self.label2id = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.sentiment_map = {
            'happy': 'positive',
            'surprise': 'positive',

            'neutral': 'neutral',

            'anger': 'negative',
            'disgust': 'negative',
            'fear': 'negative',
            'sad': 'negative',
            'contempt': 'negative',
        }

        all_sentiments = sorted(set(self.sentiment_map.values()))
        self.sentiment2id = {s: i for i, s in enumerate(all_sentiments)}
        self.id2sentiment = {v: k for k, v in self.sentiment2id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['pth'])  # 例如 affectnet/anger/image0000006.jpg
        label_str = row['label']
        emotion_label = self.label2id[label_str]
        sentiment_str = self.sentiment_map.get(label_str, 'neutral')
        sentiment_label = self.sentiment2id[sentiment_str]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, emotion_label, sentiment_label

if __name__ == "__main__":
    image_root = "/data/facial_image/affectnet"
    csv_path = os.path.join(image_root, "labels.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = AffectNetDataset(csv_path=csv_path, image_root=image_root, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 读取一个 batch 测试
    for images, emotion_labels, sentiment_labels in loader:
        print(images.shape)  # [B, 4, 224, 224]
        print(emotion_labels.shape)  # [B]
        print(sentiment_labels.shape)
        print(dataset.label2id)
        print(dataset.sentiment2id)
        break