# WGAN-GP Face Generator

A Wasserstein GAN with Gradient Penalty for generating 128x128 face images.

<p align="center">
  <img src="samples/sample_1.png" width="128" />
  <img src="samples/sample_2.png" width="128" />
  <img src="samples/sample_3.png" width="128" />
  <img src="samples/sample_4.png" width="128" />
  <img src="samples/sample_5.png" width="128" />
</p>

## Architecture

### Generator

The generator maps a 128-dimensional latent vector to a 128x128 RGB image through a series of transposed convolution layers with residual blocks.

`latent z (128,) -> reshape (128, 1, 1) -> ConvTranspose2d blocks -> tanh -> image (3, 128, 128)`

- **Transposed convolutions** (`ConvTranspose2d` with `kernel_size=2, stride=2`) progressively upsample spatial dimensions. Each layer doubles the resolution while a learned kernel controls how information is distributed across the new pixels.
- **BatchNorm + GELU** after each transposed convolution (except the final layer). BatchNorm is safe here because the gradient penalty only involves the discriminator — the generator never participates in the GP computation.
- **Residual blocks** (`layer_repeat=5`): Each upsampling stage is followed by 5 residual blocks (`Conv2d(3x3, stride=1, pad=1) -> BatchNorm -> GELU` with a skip connection). This allows deeper per-resolution feature processing without vanishing gradients.
- **`tanh` output** squashes pixel values to [-1, 1], matching the normalized training data.

### Discriminator (Critic)

The discriminator scores images with a raw scalar (the Wasserstein distance estimate) — there is no sigmoid, because the WGAN objective directly optimizes `E[D(real)] - E[D(fake)]` rather than binary cross-entropy.

`image (3, 128, 128) -> Conv2d blocks -> Flatten -> MLP -> scalar score`

- **Strided convolutions instead of MaxPool**: MaxPool has sparse, non-informative gradients — only the max element in each pooling window receives gradient; all others get zero. This would degrade the gradient penalty signal, which requires well-behaved gradients flowing back through the entire discriminator. Strided convolutions (`Conv2d(..., stride=2)`) learn the downsampling and provide smooth, informative gradients throughout.
- **InstanceNorm (not BatchNorm)**: BatchNorm computes statistics across the mini-batch, introducing inter-sample dependencies. This conflicts with the gradient penalty, which is fundamentally a per-sample constraint (the gradient norm of each interpolated sample should be ~1). InstanceNorm normalizes each sample independently (`affine=True` for learnable scale/shift), preserving correct per-sample gradient computation.
- **DropBlock2D** (`drop_prob=0.1, block_size=2`): Structured spatial dropout that drops contiguous 2x2 regions of feature maps rather than individual pixels. Standard dropout is less effective for convolutions because neighboring activations carry correlated information — dropping a single pixel is easily reconstructed from its neighbors. DropBlock forces the network to learn more spatially distributed representations.
- **GELU activations**: Smoother than ReLU/LeakyReLU with non-zero gradients everywhere, which benefits the gradient penalty computation.
- **Residual blocks** (`layer_repeat_conv=4`): 4 residual blocks per conv stage for deeper feature extraction.
- **MLP head**: After flattening, a 4-layer MLP with Dropout progressively reduces to a single scalar output. No activation on the final layer — the output is a raw Wasserstein score.

### Gradient Penalty

The gradient penalty replaces weight clipping from the original WGAN, which suffered from capacity underuse (weights pushed to clip boundaries) and exploding/vanishing gradients.

For each training step, interpolated samples are constructed between real and fake images:

```
x_hat = alpha * x_real + (1 - alpha) * x_fake,  alpha ~ U(0, 1)
```

The penalty enforces the 1-Lipschitz constraint on the critic:

```
GP = lambda * E[(||grad_x D(x_hat)||_2 - 1)^2],  lambda = 10
```

This encourages the gradient norm to be exactly 1 everywhere along the interpolation path, ensuring the critic is smooth and provides useful learning signal to the generator.

## Project Structure

```
.
├── code/
│   ├── model.py        # Generator and Discriminator architectures
│   ├── train.py        # Training loop with WGAN-GP loss
│   ├── utils.py        # Hyperparameters and configuration
│   ├── data_prep.py    # Dataset loading and preprocessing
│   └── vis_outs.py     # Generate and display samples from a trained model
├── samples/            # Sample generated faces
├── data/               # Training images (not tracked)
└── checkpoints/        # Saved model weights (not tracked)
```

## Quick Start

### Install dependencies

```bash
pip install torch torchvision matplotlib tqdm dropblock
```

### Prepare data

Place your face images (`.jpg`, `.png`, `.jpeg`) in the `data/` directory. Images are resized to 128x128 and normalized to [-1, 1].

### Train

```bash
python code/train.py
```

Checkpoints are saved every 5 epochs to `checkpoints/`. The best model (by validation Wasserstein distance) is saved to `checkpoints/best_model.pth`.

To resume from a checkpoint, set `LOAD_STATE = True` in `code/utils.py`.

### Generate samples

```bash
python code/vis_outs.py
```

Generates and displays 32 face images from the best saved checkpoint.

## Training Details

- **WGAN-GP loss**: The critic maximizes `E[D(real)] - E[D(fake)]` while the generator minimizes `-E[D(fake)]`. The gradient penalty is added to the critic loss.
- **N_CRITIC = 3**: The critic trains 3 times per generator step. The critic needs to provide a good Wasserstein distance estimate before the generator updates — training them equally would leave the critic too weak to guide the generator.
- **Separate learning rates**: `LR_D = 2e-4`, `LR_G = 1e-4`. The higher critic LR helps it stay ahead of the generator.
- **Adam optimizer**: `betas = (0.0, 0.9)`. Beta1=0 (no momentum) is standard for WGAN-GP — momentum can cause the critic to overshoot and oscillate.
- **Linear warmup**: LR ramps linearly from 0 to target over 500 steps, avoiding early instability.
- **Gradient clipping**: `max_norm = 10` for both networks as a safety net against gradient spikes.
- **Weight initialization**: `N(0, 0.02)` for all weight matrices; zeros for biases; ones/zeros for norm layers.

## Hyperparameters

| Parameter | Value |
|---|---|
| Latent dim | 128 |
| Image size | 128x128 |
| Batch size | 256 |
| Epochs | 1024 |
| LR (Generator) | 1e-4 |
| LR (Discriminator) | 2e-4 |
| Adam betas | (0.0, 0.9) |
| Weight decay | 0.0 |
| N_CRITIC | 3 |
| GP lambda | 10 |
| Gradient clip max norm | 10.0 |
| Warmup steps | 500 |
| G max channels | 512 |
| G residual blocks per layer | 5 |
| D conv layers | 4 |
| D residual blocks per conv layer | 4 |
| D MLP layers | 4 |
| D initial channels | 64 |
| DropBlock (prob, size) | (0.1, 2) |
| MLP dropout | 0.1 |
| Train/val split | 90/10 |
| Init std | 0.02 |

## License

MIT
