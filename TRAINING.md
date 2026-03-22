# Training Guide

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure a CUDA-capable GPU or Apple Silicon Mac (MPS) is available. The device is selected automatically in `code/utils.py`:

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
```

## Preparing the Dataset

Place face images (PNG, JPG, or JPEG) in the `data/` directory at the project root. The data loader will pick up all supported image files from this directory. Images are resized to 128x128 during preprocessing.

```
wgan-gp-faces/
└── data/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

## Training from Scratch

```bash
python code/train.py
```

This will:

1. Initialize the Generator and Discriminator with random weights drawn from N(0, 0.02)
2. Train for up to 1024 epochs with batch size 256
3. Save a checkpoint every 5 epochs to `checkpoints/checkpoint_epoch_{n}.pth`
4. Save the best model to `checkpoints/best_model.pth` whenever validation Wasserstein distance improves

Training prints per-epoch statistics:

```
Epoch: 0
Train:
  D Loss: -0.1234 | G Loss: -0.5678
Val:
  D Loss: -0.2345 | G Loss: -0.4567
```

The critic (D) loss approximates the negative Wasserstein distance. A more negative value generally indicates better critic separation between real and generated samples.

## Resuming Training

To resume from the best saved checkpoint, edit `code/utils.py`:

```python
LOAD_STATE = True
```

This loads the Generator, Discriminator, and both optimizer states from `checkpoints/best_model.pth`. Training continues from epoch 0 in terms of the loop counter, but model weights and optimizer momentum are restored.

## Adjusting Hyperparameters

Key parameters are defined in `code/utils.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 1024 | Total training epochs |
| `N_CRITIC` | 3 | Critic updates per generator update |
| `WARMUP_STEPS` | 500 | Linear LR warmup duration (optimizer steps) |
| `SAVE_EVERY` | 5 | Checkpoint save interval (epochs) |

Optimizer learning rates and betas:

```python
LR_G = 1e-4
LR_D = 2e-4
BETAS = (0.0, 0.9)
```

Batch size and data settings:

```python
BATCH_SIZE = 256
NUM_WORKERS = 12
```

Generator and Discriminator architecture parameters (latent dim, max channels, layer counts, residual block depth) are also configured in `code/utils.py`.

## Generating Samples

After training, generate and display face samples:

```bash
python code/vis_outs.py
```

This loads the Generator from `checkpoints/best_model.pth`, generates 64 images from random latent vectors, and displays them one at a time using matplotlib.

## Checkpoint Format

Each `.pth` file is a dictionary containing:

```python
{
    'G': Generator.state_dict(),
    'D': Discriminator.state_dict(),
    'optim_G': optim_G.state_dict(),
    'optim_D': optim_D.state_dict(),
    'epoch': int,
    'val_g_loss': float,
    'val_d_loss': float,
    'best_wasserstein': float
}
```
