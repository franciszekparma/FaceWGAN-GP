# WGAN-GP Face Generator

Wasserstein GAN with Gradient Penalty for 128x128 face image synthesis, implemented in PyTorch.

## Sample Outputs

Generated faces from the trained model:

<p align="center">
  <img src="samples/sample_1.png" width="128" alt="Generated face 1">
  <img src="samples/sample_2.png" width="128" alt="Generated face 2">
  <img src="samples/sample_3.png" width="128" alt="Generated face 3">
  <img src="samples/sample_4.png" width="128" alt="Generated face 4">
  <img src="samples/sample_5.png" width="128" alt="Generated face 5">
</p>

## Overview

This project implements a WGAN-GP for generating realistic human face images at 128x128 resolution. The architecture combines several techniques from the GAN literature: Wasserstein distance with gradient penalty as the training objective, deep residual blocks in both networks, InstanceNorm and DropBlock regularization in the critic, and a two timescale update rule for optimizer learning rates.

The generator progressively upsamples a 128-dimensional latent vector through 7 transposed convolution stages, each refined by 5 residual blocks. The discriminator (critic) encodes images through 4 convolutional stages with residual connections, then classifies through a 4-layer MLP head. The output is an unbounded scalar consistent with the Wasserstein formulation.

## Architecture

### Generator

The generator maps a latent vector `z ~ N(0, 1)` of dimension 128, reshaped to `(B, 128, 1, 1)`, to a 128x128 RGB image. It consists of 7 `GeneratorLayer` blocks, each performing a transposed convolution for spatial upsampling followed by multiple residual blocks for feature refinement.

The channel dimensions are computed dynamically: starting from `latent_dim=128`, they double to a maximum of `max_dim=512`, then halve back down. With `skip_last_layers=2`, two intermediate stages are omitted, and the final layer maps directly from 32 channels to 3 (RGB).

The resulting dimension sequence is `[128, 256, 512, 256, 128, 64, 32, 3]` across the 7 layers:

| Stage | Input Channels | Output Channels | Spatial Resolution | Upsampling | Normalization | Activation | Residual Blocks |
|-------|---------------|-----------------|-------------------|------------|---------------|------------|-----------------|
| 1 | 128 | 256 | 1x1 -> 2x2 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 2 | 256 | 512 | 2x2 -> 4x4 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 3 | 512 | 256 | 4x4 -> 8x8 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 4 | 256 | 128 | 8x8 -> 16x16 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 5 | 128 | 64 | 16x16 -> 32x32 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 6 | 64 | 32 | 32x32 -> 64x64 | ConvTranspose2d(k=2, s=2) | BatchNorm2d | GELU | 5 |
| 7 | 32 | 3 | 64x64 -> 128x128 | ConvTranspose2d(k=2, s=2) | -- | -- | 5 |

**Residual blocks (stages 1-6):** Each block applies `Conv2d(3x3, stride=1, padding=1) -> BatchNorm2d -> GELU` and adds the result back to the input via a skip connection. Bias is disabled in these convolutions since BatchNorm absorbs it.

**Residual blocks (stage 7, final):** The final stage uses a pre-activation design: `BatchNorm2d -> GELU -> Conv2d(3x3, stride=1, padding=1)` with bias enabled and skip connections. This follows the pre-activation residual block pattern from He et al. ("Identity Mappings in Deep Residual Networks", 2016), which places normalization and activation before the convolution. Moving normalization before the convolution in the output layer avoids applying BatchNorm directly to the final RGB pixel values while still allowing the residual blocks to refine the output.

The final output of the sequential is passed through **tanh**, producing pixel values in [-1, 1].

**Upsampling strategy:** All transposed convolutions use `kernel_size=2, stride=2, padding=0`. This produces a clean 2x spatial doubling without the overlapping receptive fields that occur with the more common `kernel_size=4, stride=2, padding=1` configuration. The overlap in `k=4` upsampling is a known source of checkerboard artifacts (Odena et al., "Deconvolution and Checkerboard Artifacts", 2016); using `k=2` avoids this by ensuring each output pixel is produced by exactly one input pixel.

### Discriminator (Critic)

The discriminator takes a 128x128 RGB image and outputs an unbounded scalar score. It is divided into a convolutional feature extractor and an MLP classification head.

**Convolutional section** (4 `DiscriminatorConvLayer` blocks):

| Layer | Input Channels | Output Channels | Spatial Resolution | Conv Params | Norm | Activation | Regularization | Residual Blocks |
|-------|---------------|-----------------|-------------------|-------------|------|------------|----------------|-----------------|
| 1 | 3 | 64 | 128x128 -> 64x64 | k=4, s=2, p=1 | InstanceNorm2d | GELU | DropBlock2D | 4 (skip) |
| 2 | 64 | 128 | 64x64 -> 32x32 | k=4, s=2, p=1 | InstanceNorm2d | GELU | DropBlock2D | 4 (skip) |
| 3 | 128 | 256 | 32x32 -> 16x16 | k=4, s=2, p=1 | InstanceNorm2d | GELU | DropBlock2D | 4 (skip) |
| 4 | 256 | 512 | 16x16 -> 2x2 | k=3, s=1, p=1 | -- | -- | -- | 3 (no skip) |

**Layers 1-3** each apply `Conv2d(k=4, s=2, p=1) -> InstanceNorm2d(affine=True) -> GELU -> DropBlock2D(block_size=2, drop_prob=0.1)`, followed by 4 residual blocks with skip connections. Each residual block applies `Conv2d(3x3, s=1, p=1) -> InstanceNorm2d(affine=True) -> GELU`.

**Layer 4** (last) uses a stride-1 convolution `Conv2d(k=3, s=1, p=1)` with no normalization or activation, followed by 3 sequential downsampling blocks **without** skip connections. Each block applies `InstanceNorm2d(affine=True) -> GELU -> Conv2d(k=4, s=2, p=1)`, reducing spatial dimensions by half each time (16x16 -> 8x8 -> 4x4 -> 2x2). This is again the pre-activation residual ordering (He et al., 2016), though here without the skip connection, making these blocks purely sequential downsampling stages.

The output is flattened to a vector of size `512 * 2 * 2 = 2048`.

**MLP section** (4 `DiscriminatorMLPLayer` blocks):

| Layer | Input Features | Output Features | Activation | Regularization |
|-------|---------------|-----------------|------------|----------------|
| 1 | 2048 | 1024 | GELU | Dropout(0.2) |
| 2 | 1024 | 512 | GELU | Dropout(0.1) |
| 3 | 512 | 256 | GELU | Dropout(0.1) |
| 4 | 256 | 1 | -- | -- |

The first MLP layer uses a higher dropout rate (0.2) than subsequent layers (0.1). The final layer is a bare linear projection to a single scalar with no activation, as required by the Wasserstein formulation.

### Design Choices and References

**Wasserstein distance with gradient penalty (WGAN-GP).** The training objective follows Gulrajani et al. ("Improved Training of Wasserstein GANs", NeurIPS 2017). Instead of weight clipping to enforce the Lipschitz constraint on the critic, a gradient penalty term penalizes deviations of the gradient norm from 1 on interpolated samples between real and generated data. This provides more stable gradients and a loss value that correlates with sample quality.

**InstanceNorm in the critic.** The critic uses InstanceNorm2d with learnable affine parameters instead of BatchNorm2d. The WGAN-GP paper explicitly recommends against BatchNorm in the critic because it introduces inter-sample dependencies within a batch, which conflicts with the per-sample gradient penalty computation. InstanceNorm (Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization", 2016) normalizes each sample independently, preserving the per-sample nature of the penalty. The generator retains BatchNorm2d, which is standard practice following Radford et al. (DCGAN, 2016).

**GELU activation.** Both networks use GELU (Hendrycks & Gimpel, "Gaussian Error Linear Units", 2016) throughout, replacing the ReLU/LeakyReLU activations traditionally used in GANs. GELU provides a smooth, non-monotonic activation that has shown benefits in transformer architectures and, empirically, in deep convolutional networks.

**DropBlock regularization.** The critic's convolutional layers use DropBlock2D (Ghiasi et al., "DropBlock: A Regularization Method for Convolutional Networks", NeurIPS 2018) with `block_size=2` and `drop_prob=0.1`. Unlike standard dropout, which drops individual activations, DropBlock drops contiguous spatial regions from feature maps. This is more effective for convolutional networks because adjacent activations are highly correlated, and dropping individual pixels is easily compensated by neighbors.

**Residual connections.** Both networks employ residual skip connections (He et al., "Deep Residual Learning for Image Recognition", CVPR 2016), enabling the training of substantially deeper architectures than typical GAN generators and discriminators. The generator uses 5 residual blocks per stage (35 total across 7 stages). The critic uses 4 per standard convolutional stage (12 across 3 stages).

**Pre-activation residual blocks.** The generator's final stage and the critic's last convolutional layer both place BatchNorm/InstanceNorm and GELU before the convolution (He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016), rather than the standard post-activation order.

**Two timescale update rule (TTUR).** The critic learning rate (2e-4) is set higher than the generator's (1e-4), following Heusel et al. ("GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017). This helps the critic stay ahead of the generator, providing more informative gradients.

**Adam with beta1=0.** Both optimizers use `betas=(0.0, 0.9)`, disabling the first-moment exponential moving average (momentum). This follows the recommendation from the WGAN-GP paper, which found that removing momentum improved training stability.

**Checkerboard-free upsampling.** Transposed convolutions use `k=2, s=2, p=0`, which assigns each output pixel to exactly one input pixel. The more common `k=4, s=2, p=1` creates overlapping contributions that cause checkerboard artifacts (Odena et al., "Deconvolution and Checkerboard Artifacts", Distill 2016).

## Dataset

The training data consists of 52,001 face images (PNG/JPG) stored in the `data/` directory. Preprocessing pipeline:

1. Resize to 128x128 (via `torchvision.transforms.Resize`)
2. Convert to tensor with values in [0, 1] (via `ToTensor`)
3. Normalize to [-1, 1] with `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`

The dataset is split 90/10 into training (~46,800) and validation (~5,200) sets using `torch.utils.data.random_split`.

DataLoaders use `batch_size=256`, `num_workers=12`, and `pin_memory=True`.

## Training

### Objective

The model is trained using the WGAN-GP objective:

```
L_critic = E[D(G(z))] - E[D(x_real)] + lambda * E[(||grad D(x_hat)||_2 - 1)^2]

L_generator = -E[D(G(z))]
```

where `x_hat = alpha * x_real + (1 - alpha) * G(z)` with `alpha ~ U(0, 1)`, and `lambda = 10`.

The critic is updated `N_CRITIC = 3` times for each generator update. Gradient norms are clipped to `max_norm=10.0` for both networks.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dimension | 128 |
| Image resolution | 128x128 |
| Batch size | 256 |
| Total epochs | 1024 |
| Generator learning rate | 1e-4 |
| Critic learning rate | 2e-4 |
| Optimizer | Adam |
| Adam betas | (0.0, 0.9) |
| Adam weight decay | 0.0 |
| Critic iterations per generator step | 3 |
| Gradient penalty lambda | 10 |
| Gradient clipping max norm | 10.0 |
| LR warmup steps | 500 (linear) |
| LR schedule after warmup | Constant |
| Random seed | 24 |

### Weight Initialization

All `Conv2d`, `ConvTranspose2d`, and `Linear` layers are initialized from `N(0, 0.02)` with biases set to zero. `BatchNorm2d` and `InstanceNorm2d` layers have weights initialized to 1 and biases to 0.

### Learning Rate Schedule

A linear warmup increases the learning rate from 0 to its target value over the first 500 optimizer steps. After warmup, the learning rate remains constant. The schedule is implemented via `torch.optim.lr_scheduler.LambdaLR` and is stepped per batch, not per epoch.

### Checkpointing

Checkpoints are saved every 5 epochs to `checkpoints/checkpoint_epoch_{n}.pth`. A separate `checkpoints/best_model.pth` is saved whenever the validation Wasserstein distance (estimated as the critic loss without gradient penalty on the validation set) improves. Each checkpoint contains the state dicts for both models and optimizers, the epoch number, validation losses, and the best Wasserstein distance seen so far.

## Results

Sample outputs can be generated from the best checkpoint:

```bash
python code/vis_outs.py
```

This loads `checkpoints/best_model.pth`, generates 64 face images from random latent vectors, denormalizes them from [-1, 1] to [0, 1], and displays each one using matplotlib.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA or MPS backend
- See `requirements.txt` for the full dependency list

## Installation

```bash
git clone https://github.com/<username>/wgan-gp-faces.git
cd wgan-gp-faces
pip install -r requirements.txt
```

Place face images in the `data/` directory before training.

## Usage

**Train from scratch:**

```bash
python code/train.py
```

**Resume from checkpoint:**

Set `LOAD_STATE = True` in `code/utils.py` to load weights and optimizer state from `checkpoints/best_model.pth` before continuing training.

**Generate samples:**

```bash
python code/vis_outs.py
```

See [docs/TRAINING.md](docs/TRAINING.md) for detailed training instructions.

## Checkpoints

The best-performing generator and critic weights are available for download: [best_model.pth](https://drive.google.com/file/d/1dWb7Vyx6_LDmkA5LwljK3AjRUsPOV_-F/view?usp=sharing). Place the downloaded file in the `checkpoints/` directory. Epoch-level checkpoints (`checkpoint_epoch_0.pth` through `checkpoint_epoch_85.pth`) are not included due to their size. If you need the intermediate checkpoints, contact me at **parma.franek@gmail.com**.

## Project Structure

```
.
├── code/
│   ├── model.py          # Generator and Discriminator network definitions
│   ├── train.py          # Training loop, WGAN-GP loss, gradient penalty, checkpointing
│   ├── data_prep.py      # Dataset class, preprocessing transforms, data loaders
│   ├── utils.py          # All hyperparameters (single source of truth)
│   └── vis_outs.py       # Load trained generator and visualize samples
├── samples/              # Sample generated face images
│   ├── sample_1.png
│   ├── sample_2.png
│   ├── sample_3.png
│   ├── sample_4.png
│   └── sample_5.png
├── checkpoints/          # Model checkpoints (not tracked in git)
├── docs/
│   └── TRAINING.md       # Step-by-step training guide
├── data/                 # Face image dataset (not tracked in git)
├── requirements.txt
├── .gitignore
└── LICENSE
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
