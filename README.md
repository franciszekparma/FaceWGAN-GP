# WGAN-GP Face Generator

A Wasserstein GAN with Gradient Penalty that generates 128x128 face images, built in PyTorch.

<p align="center">
  <img src="samples/sample_1.png" width="128" alt="Generated face 1">
  <img src="samples/sample_2.png" width="128" alt="Generated face 2">
  <img src="samples/sample_3.png" width="128" alt="Generated face 3">
  <img src="samples/sample_4.png" width="128" alt="Generated face 4">
  <img src="samples/sample_5.png" width="128" alt="Generated face 5">
</p>

## Key Features

- **WGAN-GP training** — gradient penalty instead of weight clipping for stable training ([Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028))
- **Deep residual architecture** — 5 residual blocks per generator stage, 4 per critic stage
- **Checkerboard-free upsampling** — `k=2, s=2` transposed convolutions avoid overlap artifacts ([Odena et al., 2016](https://distill.pub/2016/deconv-checkerboard/))
- **InstanceNorm in critic** — avoids batch-level dependencies that conflict with per-sample gradient penalty
- **DropBlock regularization** — drops contiguous spatial regions instead of individual pixels ([Ghiasi et al., 2018](https://arxiv.org/abs/1810.12890))
- **Two timescale update rule** — critic LR (2e-4) > generator LR (1e-4) for better gradient signal

## Architecture

**Generator:** Maps a 128-dim latent vector through 7 transposed-conv stages (each with 5 residual blocks) to a 128x128 RGB image. Channel sequence: `128 → 256 → 512 → 256 → 128 → 64 → 32 → 3`. Uses BatchNorm + GELU, with pre-activation residual blocks in the final stage. Output via tanh to [-1, 1].

**Critic:** 4 conv stages (with residual blocks) downsample 128x128 → 2x2, then a 4-layer MLP outputs an unbounded scalar. Uses InstanceNorm + GELU + DropBlock. No activation on the final output (Wasserstein formulation).

## Design Decisions

**No MaxPool2d** — The WGAN-GP gradient penalty requires computing gradients of the critic's output with respect to its input through `torch.autograd.grad`. MaxPool2d is a non-differentiable operation (it selects the max and drops the rest), which creates discontinuous gradients that break the penalty computation. Strided convolutions (`k=4, s=2, p=1`) achieve the same spatial downsampling while keeping the entire path smoothly differentiable.

**k=2 transposed convolutions in the generator** — The standard `k=4, s=2, p=1` transposed convolution creates overlapping receptive fields where some output pixels receive contributions from multiple input pixels. This uneven overlap is a known source of checkerboard artifacts ([Odena et al., 2016](https://distill.pub/2016/deconv-checkerboard/)). Using `k=2, s=2, p=0` gives a clean 1-to-1 mapping where each output pixel comes from exactly one input pixel, eliminating the problem at the source.

**InstanceNorm instead of BatchNorm in the critic** — The WGAN-GP gradient penalty is computed per-sample: it interpolates between individual real and fake images and penalizes each gradient independently. BatchNorm breaks this because it normalizes across the batch, making each sample's output depend on other samples in the batch. InstanceNorm normalizes each sample independently, keeping the penalty mathematically correct.

**Pre-activation residual blocks in final layers** — The generator's last stage and critic's last conv layer use the `Norm → GELU → Conv` ordering ([He et al., 2016](https://arxiv.org/abs/1603.05027)) instead of the standard `Conv → Norm → Act`. In the generator's output stage, this avoids applying BatchNorm directly to final RGB pixel values (which would distort the color distribution), while still letting the residual blocks refine the output. In the critic's last stage, the blocks intentionally have no skip connections — they act as pure sequential downsamplers (16x16 → 2x2).

**GELU over ReLU/LeakyReLU** — GELU provides a smooth, non-monotonic activation that avoids the dead neuron problem of ReLU without the arbitrary negative slope of LeakyReLU. It has shown empirical improvements in deep networks, particularly when combined with residual connections.

**Adam with beta1=0** — Disabling the first moment (momentum) follows the WGAN-GP recommendation. Momentum accumulates gradient direction over time, which can destabilize the adversarial dynamic where optimal gradients shift rapidly as the two networks co-adapt.

**DropBlock instead of Dropout in the critic** — Standard dropout drops random individual pixels, but in conv layers neighboring activations are highly correlated, so the network can easily reconstruct dropped values from their neighbors. DropBlock drops contiguous 2x2 spatial regions, forcing the network to not rely on any single local area.

**Deep residual stacking** — The generator uses 5 residual blocks per stage (35 total) and the critic uses 4 per stage (12 total). This is significantly deeper than typical GAN architectures. The residual connections make this depth trainable by providing gradient shortcuts, letting each block learn small refinements rather than full transformations.

## Training

### Objective

```
L_critic = E[D(G(z))] - E[D(x)] + λ·E[(‖∇D(x̂)‖₂ - 1)²]
L_generator = -E[D(G(z))]
```

where `x̂ = α·x + (1-α)·G(z)`, `α ~ U(0,1)`, `λ = 10`. The critic is updated 3 times per generator step.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 128 |
| Batch size | 256 |
| Epochs | 1024 |
| Generator LR | 1e-4 |
| Critic LR | 2e-4 |
| Adam betas | (0.0, 0.9) |
| Gradient penalty λ | 10 |
| Critic steps per G step | 3 |
| LR warmup | 500 steps (linear) |
| Grad clip max norm | 10.0 |

### Training Cost

Trained on an NVIDIA H100 for ~14 hours, total compute cost ~$45.

### Dataset

52k face images in `data/`, resized to 128x128 and normalized to [-1, 1]. Split 90/10 train/val.

## Quick Start

```bash
git clone https://github.com/<username>/wgan-gp-faces.git
cd wgan-gp-faces
pip install -r requirements.txt
```

Place face images in `data/`, then:

```bash
# Train
python code/train.py

# Resume from checkpoint (set LOAD_STATE = True in code/utils.py)
python code/train.py

# Generate samples
python code/vis_outs.py
```

See [docs/TRAINING.md](docs/TRAINING.md) for a detailed training guide.

## Pretrained Weights

Download [best_model.pth](https://drive.google.com/file/d/1dWb7Vyx6_LDmkA5LwljK3AjRUsPOV_-F/view?usp=sharing) and place it in `checkpoints/`. For intermediate epoch checkpoints, contact **parma.franek@gmail.com**.

## Project Structure

```
├── code/
│   ├── model.py       # Generator & Critic definitions
│   ├── train.py       # Training loop, WGAN-GP loss, checkpointing
│   ├── data_prep.py   # Dataset, transforms, data loaders
│   ├── utils.py       # All hyperparameters
│   └── vis_outs.py    # Generate & visualize samples
├── samples/           # Sample outputs
├── checkpoints/       # Saved models (not in git)
├── data/              # Training images (not in git)
├── docs/TRAINING.md   # Training guide
├── requirements.txt
└── LICENSE
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA or MPS)
- See `requirements.txt`

## License

MIT — see [LICENSE](LICENSE).
