from pathlib import Path
from typing import Optional, List, Tuple, Any

from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class DogsData(Dataset):
    def __init__(self, root: str, transform: Optional[Any] = None, identities: Optional[List[str]] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.identity_to_idx = {}

        #grab all the dog folders
        folders = sorted([d for d in self.root.iterdir() if d.is_dir()])

        if identities is not None:
            folders = [d for d in folders if d.name in identities]

        for idx, folder in enumerate(folders):
            self.identity_to_idx[folder.name] = idx
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_path), idx, folder.name))

        self.idx_to_identity = {v: k for k, v in self.identity_to_idx.items()}
        #print(f"loaded {len(self.samples)} images from {len(self.identity_to_idx)} dogs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str]:
        path, label, identity = self.samples[idx]

        try:
            image = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print("problem getting, skip it.")
            return self.__getitem__((idx+1)%len(self))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label, identity


def get_train_transforms(size: int = 224) -> Any:
    #pretty strong augs so it handles different angles better
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.8),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = 224) -> Any:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2(),
    ])