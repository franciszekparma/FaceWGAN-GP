# WGAN-GP Face Generator

A from-scratch PyTorch implementation of a Wasserstein GAN with Gradient Penalty that learns to generate 128x128 human faces from noise.

<p align="center">
  <img src="samples/sample_1.png" width="200" />
  <img src="samples/sample_2.png" width="200" />
  <img src="samples/sample_3.png" width="200" />
  <img src="samples/sample_4.png" width="200" />
  <img src="samples/sample_5.png" width="200" />
</p>

<p align="center"><i>Selected outputs from the trained generator. Each image is synthesized from a random 128-d latent vector.</i></p>

---

## Why WGAN-GP?

Vanilla GANs minimize the Jensen-Shannon divergence between real and generated distributions. When those distributions have little overlap (which is almost always the case in high-dimensional image space), JS divergence saturates and the generator receives near-zero gradients. Training stalls.

The **Wasserstein distance** (Earth Mover's Distance) doesn't have this problem — it provides a smooth, meaningful gradient signal even when distributions are far apart. But computing it requires the critic to be **1-Lipschitz**, which the original WGAN enforced via crude weight clipping.

**Gradient Penalty** replaces weight clipping with a soft constraint: sample points along straight lines between real and fake images, and penalize the critic whenever its gradient norm deviates from 1. This gives stable training without crippling the critic's capacity.

```
Standard GAN loss:     min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
                       ↑ saturates when D is confident

WGAN-GP loss:          min_G max_D  E[D(x)] - E[D(G(z))] - λ·E[(‖∇D(x̂)‖₂ - 1)²]
                       ↑ linear in D's output → always provides gradient
```

---

## How It Works

### The Generator

Takes a 128-dimensional noise vector and progressively upsamples it into a 128x128 RGB image.

Each upsampling stage doubles the spatial resolution via `ConvTranspose2d(kernel=2, stride=2)`, followed by **5 residual blocks** that refine details at that resolution before moving on. BatchNorm is used freely here since the gradient penalty only constrains the critic.

```
z ∈ R^128  →  reshape to (128, 1, 1)

    ↓  ConvTranspose2d + BN + GELU + 5×ResBlock     1×1   → 2×2
    ↓  ConvTranspose2d + BN + GELU + 5×ResBlock     2×2   → 4×4
    ↓  ConvTranspose2d + BN + GELU + 5×ResBlock     4×4   → 8×8
    ↓  ConvTranspose2d + BN + GELU + 5×ResBlock     8×8   → 16×16
    ↓  ConvTranspose2d + BN + GELU + 5×ResBlock     16×16 → 32×32
    ↓  ConvTranspose2d + BN + GELU                  32×32 → 64×64
    ↓  ConvTranspose2d → tanh                       64×64 → 128×128

Output: image ∈ [-1, 1]^(3×128×128)
```

### The Critic

Scores an image with a single unbounded scalar — higher means "more real." There is no sigmoid because the Wasserstein objective needs the output to be an unconstrained real number, not a probability.

The critic has several design choices driven specifically by the gradient penalty:

**InstanceNorm instead of BatchNorm.** The GP computes `∂D(x̂)/∂x̂` per-sample. BatchNorm couples samples through shared batch statistics, corrupting these per-sample gradients. InstanceNorm normalizes over spatial dimensions only, keeping each sample independent.

**Strided convolutions instead of MaxPool.** MaxPool passes gradients only through the max element in each window — everyone else gets zero. This creates sparse, uninformative gradients that degrade the GP signal. Learned strided convolutions provide dense, smooth gradients throughout.

**DropBlock2D instead of standard Dropout.** In convolutional layers, neighboring activations are highly correlated. Dropping individual pixels does almost nothing because neighbors reconstruct the missing values. DropBlock drops contiguous 2x2 spatial regions, forcing genuinely distributed representations.

```
image (3, 128, 128)
    ↓  Conv2d(s=2) + IN + GELU + DropBlock + 4×ResBlock     → (64,  64, 64)
    ↓  Conv2d(s=2) + IN + GELU + DropBlock + 4×ResBlock     → (128, 32, 32)
    ↓  Conv2d(s=2) + IN + GELU + DropBlock + 4×ResBlock     → (256, 16, 16)
    ↓  Conv2d(s=1) + 3×ResBlock (no skip, downsample)       → (512,  2,  2)
    ↓  Flatten                                               → (2048,)
    ↓  Linear + GELU + Dropout(0.2)
    ↓  Linear + GELU + Dropout(0.1)
    ↓  Linear + GELU + Dropout(0.1)
    ↓  Linear → scalar
```

### The Gradient Penalty

For each training step, interpolated samples are constructed between real and fake images, and the critic's gradient norm on those samples is penalized toward 1:

```python
alpha = torch.rand((B, 1, 1, 1), device=device)
x_hat = alpha * x_real + (1 - alpha) * x_fake          # interpolate

score = discriminator(x_hat)
grads = torch.autograd.grad(outputs=score, inputs=x_hat, ...)
norm = grads.view(B, -1).norm(2, dim=1)

GP = lambda * ((norm - 1) ** 2).mean()                # penalize deviation from 1
```

### Training Loop

The critic trains **3 steps** for every generator step. It needs to be a good Wasserstein distance estimator before the generator can get a useful learning signal from it.

```
for each batch of real images:
    repeat 3 times:
        generate fakes
        D_loss = E[D(fake)] - E[D(real)] + GP(real, fake)
        update critic

    generate fakes
    G_loss = -E[D(fake)]
    update generator
```

Both networks use Adam with `betas=(0.0, 0.9)` — zero momentum is standard for WGAN-GP because momentum can cause the critic to overshoot and oscillate. Learning rates warm up linearly from 0 over 500 steps.

---

## Project Structure

```
.
├── code/
│   ├── model.py        # Generator and Discriminator architectures
│   ├── train.py        # WGAN-GP training loop, gradient penalty, validation
│   ├── utils.py        # All hyperparameters and paths
│   ├── data_prep.py    # Dataset class, transforms, train/val split
│   └── vis_outs.py     # Load checkpoint and display generated faces
├── docs/
│   └── TRAINING.md     # Detailed training and checkpointing guide
├── data/               # Face images — not tracked
├── checkpoints/        # Model weights — not tracked
├── samples/            # Sample outputs
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/franciszekparma/wgan-gp-faces.git
cd wgan-gp-faces
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `tqdm`, `dropblock`

### Prepare data

Drop face images (`.jpg`, `.png`, `.jpeg`) into `data/`. They'll be resized to 128x128 and normalized to [-1, 1] automatically.

### Train

```bash
python code/train.py
```

Checkpoints save every 5 epochs. The best model (lowest validation Wasserstein distance) is saved to `checkpoints/best_model.pth`. To resume from a checkpoint, set `LOAD_STATE = True` in `code/utils.py`.

### Generate faces

```bash
python code/vis_outs.py
```

---

## Hyperparameters

Everything lives in [`code/utils.py`](code/utils.py).

**Training**
| | |
|---|---|
| Batch size | 256 |
| Epochs | 1024 |
| Critic steps per generator step | 3 |
| LR (generator) | 1e-4 |
| LR (critic) | 2e-4 |
| Adam betas | (0.0, 0.9) |
| GP lambda | 10 |
| Gradient clip | 10.0 |
| Warmup steps | 500 |
| Train/val split | 90/10 |
| Weight init | N(0, 0.02) |

**Generator**
| | |
|---|---|
| Latent dim | 128 |
| Max channels | 512 |
| Residual blocks per stage | 5 |

**Critic**
| | |
|---|---|
| Conv stages | 4 |
| Residual blocks per conv stage | 4 |
| Initial channels | 64 |
| MLP layers | 4 |
| DropBlock (prob, size) | (0.1, 2) |
| MLP dropout | 0.1 / 0.2 (first layer) |

---

## Hardware

Trained on an rented H100.
* Total time: 14h
* Money spend: 45$  

---

## References

- Goodfellow et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- Arjovsky et al. (2017). [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- Gulrajani et al. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- Ghiasi et al. (2018). [DropBlock: A Regularization Method for Convolutional Networks](https://arxiv.org/abs/1810.12890)

---

## License

MIT &copy; franciszekparma
