"""
📂 CCTV 이상행동 데이터셋 (전처리된 프레임 기반)
- make_processed_frames.py 로 생성된 processed_frames 를 읽어옴
"""

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms


class CCTVDataset(Dataset):
    """CCTV 이상행동 데이터셋 (processed_frames 기반)"""

    def __init__(self, root_dir, split="train", num_frames=16, img_size=224):
        """
        Args:
            root_dir: datasets/ 경로
            split: 'train' or 'val'
            num_frames: 추출할 프레임 수
            img_size: 입력 이미지 크기
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size

        # 클래스 정의 (영문명 기준)
        self.classes = ["crowd", "fight", "fall"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # processed_frames 경로
        self.frame_root = self.root_dir / "processed_frames" / self.split
        self.videos = self._load_preprocessed()

        print(f"✅ {split.upper()}: {len(self.videos)}개 영상 (전처리 프레임 기반)")

        # 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_preprocessed(self):
        """processed_frames/{split}/{class}/ 내 모든 영상 폴더 로드"""
        videos = []
        for class_name in self.classes:
            class_dir = self.frame_root / class_name
            if not class_dir.exists():
                print(f"⚠️ Missing class dir: {class_dir}")
                continue

            for video_dir in sorted(class_dir.iterdir()):
                if video_dir.is_dir():
                    videos.append({
                        "class": class_name,
                        "video": video_dir.name
                    })
        return videos

    def _sample_frames(self, frame_dir):
        """프레임 폴더에서 num_frames만큼 균등 샘플링"""
        frames = sorted(frame_dir.glob("*.jpg"))
        total = len(frames)
        if total == 0:
            raise ValueError(f"❌ No frames found in {frame_dir}")

        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            selected = [frames[i] for i in indices]
        else:
            # 부족하면 반복해서 채움
            repeat_factor = int(np.ceil(self.num_frames / total))
            selected = (frames * repeat_factor)[:self.num_frames]

        return selected

    def __getitem__(self, idx):
        info = self.videos[idx]
        class_name = info["class"]
        label = self.class_to_idx[class_name]
        frame_dir = self.frame_root / class_name / info["video"]

        frame_paths = self._sample_frames(frame_dir)
        frames = []

        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            frames.append(img)

        if len(frames) == 0:
            raise ValueError(f"❌ No valid frames in {frame_dir}")

        frames = torch.stack(frames)
        return frames, label

    def __len__(self):
        return len(self.videos)
