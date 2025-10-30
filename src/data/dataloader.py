from torch.utils.data import DataLoader
from src.data.dataset import CCTVDataset

def get_dataloaders(batch_size=8, num_frames=16, num_workers=0, img_size=224):
    train_dataset = CCTVDataset(
        root_dir="datasets",
        split="train",
        num_frames=num_frames,
        img_size=img_size,
    )
    val_dataset = CCTVDataset(
        root_dir="datasets",
        split="val",
        num_frames=num_frames,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, val_loader
