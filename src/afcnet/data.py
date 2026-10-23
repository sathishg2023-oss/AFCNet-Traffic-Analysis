"""UA-DETRAC image/annotation pairing and transforms."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def make_transform(train: bool = False, image_size: int = 224):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1)]
    ops += [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)

def pair_files(image_dir: str | Path, label_dir: str | Path):
    """Pair files by relative path/stem; fail on ambiguous duplicate stems."""
    image_root, label_root = Path(image_dir), Path(label_dir)
    if not image_root.is_dir(): raise FileNotFoundError(f'Image directory not found: {image_root}')
    if not label_root.is_dir(): raise FileNotFoundError(f'Annotation directory not found: {label_root}')
    images = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(label_root.rglob('*.txt'))
    label_by_rel = {(p.relative_to(label_root).with_suffix('').as_posix()): p for p in labels}
    label_by_stem = {}
    for p in labels: label_by_stem.setdefault(p.stem, []).append(p)
    pairs, missing = [], []
    for img in images:
        rel = img.relative_to(image_root).with_suffix('').as_posix()
        lab = label_by_rel.get(rel)
        if lab is None:
            candidates = label_by_stem.get(img.stem, [])
            if len(candidates) == 1: lab = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(f'Ambiguous annotation for {img}: {[str(p) for p in candidates]}')
        if lab is None: missing.append(img)
        else: pairs.append((img, lab))
    return pairs, missing

class VehicleCountDataset(Dataset):
    """Count non-empty annotation lines; valid only if each line is one vehicle."""
    def __init__(self, image_dir, label_dir, train=False, image_size=224, exclude_empty_annotations=False):
        pairs, missing = pair_files(image_dir, label_dir)
        self.unmatched_images = missing
        self.pairs = []
        for img, label in pairs:
            lines = [line for line in label.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]
            if exclude_empty_annotations and not lines: continue
            self.pairs.append((img, label, len(lines)))
        self.transform = make_transform(train, image_size)
        if not self.pairs: raise ValueError('No matched image/annotation pairs were found.')
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, label_path, count = self.pairs[idx]
        with Image.open(img_path) as im: image = im.convert('RGB')
        return self.transform(image), torch.tensor(float(count), dtype=torch.float32)
