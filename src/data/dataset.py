"""
전처리 기반 CCTV 이상행동 데이터셋
(BASELINE/src/data/dataset.py)
"""

import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class CCTVDataset(Dataset):
    """CCTV 이상행동 데이터셋 (전처리된 프레임 기반)"""

    def __init__(self, root_dir, split='train', num_frames=16, img_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        
        self.classes = ['군집', '싸움', '쓰러짐']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # processed_frames 기준으로 데이터 구성
        self.frame_root = self.root_dir / "processed_frames" / self.split
        self.videos = self._load_preprocessed()

        print(f"✅ {split.upper()}: {len(self.videos)}개 영상 (전처리 프레임 기반)")

    def _load_preprocessed(self):
        """전처리된 프레임 폴더 구조 읽기"""
        videos = []
        for class_name in self.classes:
            class_path = self.frame_root / class_name
            if not class_path.exists():
                continue

            for video_dir in class_path.iterdir():
                if not video_dir.is_dir():
                    continue
                frame_files = sorted(video_dir.glob("frame_*.jpg"))
                if len(frame_files) == 0:
                    continue
                videos.append({
                    'frames': frame_files,
                    'label': self.class_to_idx[class_name],
                    'video_name': video_dir.name,
                    'class_name': class_name
                })
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frame_files = video_info['frames']

        # 균등 샘플링
        if len(frame_files) < self.num_frames:
            indices = np.random.choice(len(frame_files), self.num_frames, replace=True)
        else:
            indices = np.linspace(0, len(frame_files)-1, self.num_frames, dtype=int)

        selected_frames = [frame_files[i] for i in indices]

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img)

        video_tensor = torch.stack(frames)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std

        return video_tensor, video_info['label']


# ============================================================
# 🔧 전처리 함수: mp4 → 프레임 JPG
# ============================================================

def preprocess_videos(root_dir, split='train', num_frames=16, use_json=True):
    root = Path(root_dir)
    video_root = root / split
    output_root = root / "processed_frames" / split
    json_root = root / "labels_json"

    classes = ['군집', '싸움', '쓰러짐']

    for class_name in classes:
        class_path = video_root / class_name
        output_class_path = output_root / class_name
        output_class_path.mkdir(parents=True, exist_ok=True)

        for video_file in sorted(class_path.glob("*.mp4")):
            video_name = video_file.stem
            save_dir = output_class_path / video_name
            save_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(str(video_file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start, end = 0, total_frames - 1

            # JSON 구간 처리
            if use_json:
                json_path = json_root / class_name / f"{video_name}.json"
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        ann = json_data.get("annotations", {})
                        event_frame = ann.get("event_frame", [])
                        if event_frame and len(event_frame) > 0:
                            start, end = event_frame[0]
                            end = min(end, total_frames - 1)

            indices = np.linspace(start, end, num_frames, dtype=int)

            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = save_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            cap.release()

            print(f"📸 {class_name}/{video_name} → {len(indices)} 프레임 저장 완료")

    print(f"\n✅ 모든 {split} 데이터 전처리 완료!")

if __name__ == "__main__":
    # 전처리 실행
    preprocess_videos("datasets", split="train", num_frames=16)
    preprocess_videos("datasets", split="val", num_frames=16)
