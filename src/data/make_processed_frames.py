"""
🎬 JSON 기반 프레임 추출 스크립트 (train/val 모두 처리)
- datasets/{train,val}/<class>/*.mp4  원본 영상에서
- labels_json/<class>/*.json 의 event_frame 구간만 추출
- processed_frames/{train,val}/<class>/ 에 저장
"""

import cv2
import json
from pathlib import Path
from tqdm import tqdm


def extract_event_frames(video_path: Path, save_dir: Path, event_frames: list):
    """하나의 영상에서 이벤트 프레임 구간만 추출"""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"⚠️ 영상 읽기 실패: {video_path}")
        return

    for i, (start, end) in enumerate(event_frames):
        start = max(0, int(start))
        end = min(total - 1, int(end))
        if start >= end:
            print(f"⚠️ 이벤트 범위 이상: {start}~{end} ({video_path.name})")
            continue

        save_path = save_dir / f"{video_path.stem}_ev{i+1}"
        save_path.mkdir(parents=True, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start

        while frame_idx <= end:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = save_path / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            frame_idx += 1

    cap.release()


def make_processed_frames():
    """train / val 각각에 대해 JSON 구간 기반 프레임 생성"""
    root = Path("datasets")
    json_root = root / "labels_json"             # JSON 라벨
    save_root = root / "processed_frames"        # 출력 경로
    save_root.mkdir(parents=True, exist_ok=True)

    classes = ["crowd", "fight", "fall"]
    splits = ["train", "val"]

    for split in splits:
        print(f"\n========== {split.upper()} ==========")
        video_root = root / split
        for cls in classes:
            cls_video_dir = video_root / cls
            cls_json_dir = json_root / cls
            cls_save_dir = save_root / split / cls
            cls_save_dir.mkdir(parents=True, exist_ok=True)

            json_files = sorted(cls_json_dir.glob("*.json"))
            if not json_files:
                print(f"⚠️ JSON 없음: {cls_json_dir}")
                continue

            for json_file in tqdm(json_files, desc=f"{split}-{cls}"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                video_name = Path(data["metadata"]["file_name"]).stem
                video_path = cls_video_dir / f"{video_name}.mp4"
                if not video_path.exists():
                    continue  # val에 없는 경우는 건너뜀

                event_frames = data["annotations"].get("event_frame", [])
                if not event_frames:
                    print(f"⚠️ event_frame 없음: {json_file.name}")
                    continue

                extract_event_frames(video_path, cls_save_dir, event_frames)

    print("\n✅ 모든 train/val 이벤트 프레임 추출 완료!")


if __name__ == "__main__":
    make_processed_frames()
