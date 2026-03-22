import torch
import torchvision.transforms as TT
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os

from utils import BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT, DATA_DIR, IMAGE_SHAPE


class ImageDataset(Dataset):
  def __init__(self, root, transform):
    self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".jpg", ".png", ".jpeg"))]
    self.transform = transform

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    img = Image.open(self.paths[idx]).convert("RGB")
    return self.transform(img)


transforms = TT.Compose([
    TT.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
    TT.ToTensor(),
    TT.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = ImageDataset(DATA_DIR, transforms)

n = len(dataset)
train_n = int(TRAIN_SPLIT * n)
val_n = n - train_n

train_ds, val_ds = random_split(dataset, [train_n, val_n])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)