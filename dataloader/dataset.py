import csv
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _label_from_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.suffix.lower() not in IMG_EXTS:
        raise ValueError(f"Unsupported image extension: {image_path}")
    return path.parent.name


def read_split_csv(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    class_names = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "image_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            split = row["split"].strip()
            image_path = row["image_path"].strip()
            if not split or not image_path:
                continue
            label_name = _label_from_image_path(image_path)
            rows.append({"split": split, "image_path": image_path, "label_name": label_name})
            class_names.add(label_name)
    return rows, sorted(class_names)


class CsvImageClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str,
        class_names: List[str],
        augment: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.split = split
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        all_rows, _ = read_split_csv(csv_path)
        self.samples = [row for row in all_rows if row["split"] == split]
        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {csv_path}")
        self.transform = train_transform if augment else val_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[sample["label_name"]]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
        }
