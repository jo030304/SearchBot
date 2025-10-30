"""
학습 스크립트 (MPS 안정화 버전)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import get_dataloaders
from src.models.backbones import build_model


def get_device():
    """CUDA > MPS > CPU 자동 선택"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🖥️ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("🖥️ Using CPU")
    return device


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_acc = 0.0

    def train(self, epochs, save_dir="checkpoints", backbone="model"):
        os.makedirs(save_dir, exist_ok=True)

        # ✅ 학습 히스토리 기록용 리스트
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(1, epochs + 1):
            print(f"\n📅 Epoch {epoch}/{epochs}")
            print("-" * 60)

            # ---------- Train ----------
            self.model.train()
            running_loss, correct, total = 0, 0, 0

            for imgs, labels in tqdm(self.train_loader, desc="Train", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 🔥 MPS 메모리 캐시 비우기
                if self.device.type == "mps":
                    torch.mps.empty_cache()

            train_loss = running_loss / total
            train_acc = correct / total * 100

            # ---------- Validation ----------
            val_loss, val_acc = self.evaluate()

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

            # ✅ 히스토리 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # ---------- Save Best ----------
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                best_path = Path(save_dir) / f"best_{backbone}_epoch{epoch}.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"💾 Saved best model: {best_path.name}")

            # 🔥 epoch 후에도 캐시 비우기
            if self.device.type == "mps":
                torch.mps.empty_cache()

        print(f"\n✅ Training Complete! Best Val Acc: {self.best_acc:.2f}%")

        # =========================================
        # ✅ 학습 히스토리 저장 (compare_models용)
        # =========================================
        history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accuracies,
            "val_acc": val_accuracies
        }

        history_path = Path(save_dir) / f"{backbone}_history.json"
        with open(history_path, "w") as f:
            import json
            json.dump(history, f, indent=4)
        print(f"💾 학습 히스토리 저장 완료: {history_path}")


    def evaluate(self):
        self.model.eval()
        running_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Val", leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total * 100
        return val_loss, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=0,  # macOS 필수
    )

    model = build_model(args.backbone, pretrained=True, num_classes=3)
    print(f"✅ {args.backbone.upper()} 생성 완료")

    trainer = Trainer(model, device, train_loader, val_loader, lr=args.lr)
    trainer.train(epochs=args.epochs, backbone=args.backbone)
