"""
학습 스크립트 (CLI 기반 베이스라인)
(BASELINE/scripts/train.py)
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------------------------
# 내부 모듈 import
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.dataset import CCTVDataset
from src.models.backbones import build_model, MODEL_INFO


# -------------------------------------------------
# 디바이스 자동 선택
# -------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print(f"🖥️ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("🖥️ Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    else:
        print("🖥️ Using CPU")
        return torch.device("cpu")


# -------------------------------------------------
# 학습 루프
# -------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for videos, labels in tqdm(dataloader, desc="Train", leave=False):
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


# -------------------------------------------------
# 검증 루프
# -------------------------------------------------
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Val", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc


# -------------------------------------------------
# 메인 실행
# -------------------------------------------------
def main():
    # ===============================
    # CLI 인자 설정
    # ===============================
    parser = argparse.ArgumentParser(description="Train Baseline Video Classifier")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=list(MODEL_INFO.keys()),
                        help="사용할 백본 모델 (예: resnet18, resnet50, efficientnet_b0, mobilenet_v3, convnext_tiny)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--num_frames", type=int, default=16, help="샘플링할 프레임 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률 (기본값: 1e-4)")
    parser.add_argument("--img_size", type=int, default=224, help="프레임 이미지 크기 (기본값: 224)")
    args = parser.parse_args()

    # ===============================
    # 기본 설정
    # ===============================
    root = Path(__file__).resolve().parent.parent / "datasets"
    device = get_device()

    print("\n" + "=" * 80)
    print(f"🎬 Baseline Training 시작")
    print(f"   Backbone    : {args.backbone}")
    print(f"   Epochs      : {args.epochs}")
    print(f"   Batch Size  : {args.batch_size}")
    print(f"   Num Frames  : {args.num_frames}")
    print(f"   Learning Rate: {args.lr}")
    print("=" * 80 + "\n")

    # ===============================
    # Dataset & Dataloader
    # ===============================
    train_dataset = CCTVDataset(root, split="train",
                                num_frames=args.num_frames,
                                img_size=args.img_size)
    val_dataset = CCTVDataset(root, split="val",
                              num_frames=args.num_frames,
                              img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    print(f"📂 TRAIN 샘플: {len(train_dataset)} / VAL 샘플: {len(val_dataset)}")

    # ===============================
    # 모델 생성 (backbones.py)
    # ===============================
    model = build_model(args.backbone, num_classes=3, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = 0.0

    # ===============================
    # 학습 루프
    # ===============================
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        print("-" * 60)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # 최고 정확도 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = ckpt_dir / f"best_{args.backbone}_epoch{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model: {save_path.name}")

    print(f"\n✅ Training Complete! Best Val Acc: {best_acc:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
