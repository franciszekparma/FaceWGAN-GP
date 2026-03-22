import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator
from utils import DEVICE, SHOW_BATCH_SIZE, BEST_MODEL_PATH


def main():
  G = Generator(DEVICE)

  checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
  G.load_state_dict(checkpoint['G'])
  G.to(DEVICE)
  G.eval()

  with torch.no_grad():
    out = G(SHOW_BATCH_SIZE).cpu().numpy()

  for img in out:
    img = (img + 1) / 2
    plt.imshow(img.squeeze().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
  main()