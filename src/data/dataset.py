import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random
from torchvision import transforms


class CCTVDataset(Dataset):
    """CCTV abnormal behavior dataset (supports JSON labels or preprocessed frames)"""
    
    def __init__(self, root_dir, split='train', num_frames=16, img_size=224, use_json=True):
        """
        Args:
            root_dir: datasets/ path
            split: 'train' or 'val'
            num_frames: number of frames to sample
            img_size: target image size
            use_json: whether to use JSON label data
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.use_json = use_json

        self.classes = ['crowd', 'fight', 'fall']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Frame & label paths
        self.frame_root = self.root_dir / "processed_frames" / self.split
        self.label_root = self.root_dir / "labels_json" / self.split

        # Load video metadata
        if self.use_json:
            self.videos = self._load_from_json()
        else:
            self.videos = self._load_preprocessed()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"✅ {split.upper()}: {len(self.videos)} videos (use_json={self.use_json})")


    def _load_from_json(self):
        """labels_json을 기반으로 이벤트 구간만 로드 (E04_001.mp4 → E04_001 폴더 매칭)"""
        videos = []
        for class_name in self.classes:
            class_dir = self.frame_root / class_name
            label_dir = self.label_root / class_name
            if not class_dir.exists() or not label_dir.exists():
                continue

            for json_file in label_dir.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 🎬 metadata에서 파일명 추출
                video_name = Path(data["metadata"]["file_name"]).stem  # E04_001.mp4 → E04_001
                video_dir = class_dir / video_name

                # ⚠️ 실제 프레임 폴더가 존재하는 경우만 추가
                if not video_dir.exists():
                    continue

                # 📑 event_frame 정보 읽기
                event_frames = data["annotations"].get("event_frame", [])
                for start_f, end_f in event_frames:
                    videos.append({
                        "class": class_name,
                        "video": video_name,
                        "start": int(start_f),
                        "end": int(end_f)
                    })

        print(f"📁 {self.split.upper()} loaded {len(videos)} labeled clips (use_json=True)")
        return videos


    def _load_preprocessed(self):
        """Fallback: load from processed_frames without JSON labels"""
        videos = []
        for class_name in self.classes:
            class_dir = self.frame_root / class_name
            if not class_dir.exists():
                continue
            for video_dir in class_dir.iterdir():
                if video_dir.is_dir():
                    videos.append({
                        "class": class_name,
                        "video": video_dir.name,
                        "start": None,
                        "end": None
                    })
        return videos


    def _sample_frames(self, frame_dir, start_f=None, end_f=None):
        """Uniformly sample frames within event segment"""
        frames = sorted(frame_dir.glob("*.jpg"))
        total = len(frames)
        if total == 0:
            raise ValueError(f"❌ No frames found in {frame_dir}")

        if start_f is not None and end_f is not None:
            frames = frames[start_f:end_f+1]
        if len(frames) < self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)

        selected = [frames[i] for i in indices]
        return selected


    def __len__(self):
        return len(self.videos)


    def __getitem__(self, idx):
        info = self.videos[idx]
        class_name = info["class"]
        label = self.class_to_idx[class_name]

        frame_dir = self.frame_root / class_name / info["video"]
        frame_paths = self._sample_frames(frame_dir, info["start"], info["end"])

        frames = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # [T, C, H, W]
        return frames, label
