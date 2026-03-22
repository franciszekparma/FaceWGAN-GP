# WGAN-GP Face Generator

A Wasserstein GAN with Gradient Penalty that generates 128x128 face images, built in PyTorch.

Trained on an NVIDIA H100 for 14 hours (~$45 total compute).

<p align="center">
  <img src="samples/sample_1.png" width="128" alt="Generated face 1">
  <img src="samples/sample_2.png" width="128" alt="Generated face 2">
  <img src="samples/sample_3.png" width="128" alt="Generated face 3">
  <img src="samples/sample_4.png" width="128" alt="Generated face 4">
  <img src="samples/sample_5.png" width="128" alt="Generated face 5">
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Training](#training)
- [Pretrained Weights](#pretrained-weights)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Quick Start

```bash
git clone https://github.com/<username>/wgan-gp-faces.git
cd wgan-gp-faces
pip install -r requirements.txt
```

Place face images in `data/`, then:

```bash
python code/train.py       # train from scratch
python code/vis_outs.py    # generate samples from checkpoint
```

To resume from a checkpoint, set `LOAD_STATE = True` in `code/utils.py`.

For more details see [docs/TRAINING.md](docs/TRAINING.md).

---

## Architecture

### Generator

Maps a 128-dim latent vector through 7 transposed-conv stages to a 128x128 RGB image.
Each stage contains 5 residual blocks for feature refinement.

**Channel progression:** `128 → 256 → 512 → 256 → 128 → 64 → 32 → 3`

- Stages 1-6: BatchNorm + GELU activation
- Stage 7 (final): pre-activation residual blocks (`BatchNorm → GELU → Conv`) to avoid normalizing RGB pixel values directly
- Output through tanh to [-1, 1]

### Critic

4 convolutional stages with residual blocks downsample from 128x128 to 2x2, followed by a 4-layer MLP outputting a single unbounded scalar.

- Conv stages: InstanceNorm + GELU + DropBlock
- MLP: Dropout (0.2 first layer, 0.1 rest)
- No activation on the final output (Wasserstein formulation)

---

## Design Decisions

### Why no MaxPool2d?

The WGAN-GP gradient penalty computes gradients of the critic's output with respect to its input via `torch.autograd.grad`. MaxPool2d is non-differentiable: it selects the max and discards the rest, producing discontinuous gradients that break the penalty computation. Strided convolutions (`k=4, s=2, p=1`) achieve the same spatial downsampling while keeping the entire path smoothly differentiable.

### Why k=2 transposed convolutions?

The standard `k=4, s=2, p=1` transposed convolution creates overlapping receptive fields where some output pixels receive contributions from multiple input pixels. This uneven overlap is a known source of checkerboard artifacts ([Odena et al., 2016](https://distill.pub/2016/deconv-checkerboard/)). Using `k=2, s=2, p=0` gives a clean 1-to-1 mapping where each output pixel comes from exactly one input pixel.

### Why InstanceNorm instead of BatchNorm in the critic?

The gradient penalty is computed per-sample: it interpolates between individual real and fake images and penalizes each gradient independently. BatchNorm normalizes across the batch, making each sample's output depend on other samples. InstanceNorm normalizes each sample independently, keeping the penalty mathematically correct.

### Why pre-activation residual blocks in the final layers?

The generator's last stage and critic's last conv layer use `Norm → Activation → Conv` ordering ([He et al., 2016](https://arxiv.org/abs/1603.05027)) instead of `Conv → Norm → Activation`. In the generator this avoids applying BatchNorm directly to final RGB values, which would distort the color distribution. In the critic's last stage the blocks have no skip connections and act as pure sequential downsamplers (16x16 to 2x2).

### Why GELU?

GELU provides a smooth, non-monotonic activation. It avoids the dead neuron problem of ReLU without the arbitrary negative slope of LeakyReLU.

### Why Adam with beta1=0?

Disabling momentum follows the WGAN-GP paper's recommendation. Momentum accumulates gradient direction over time, which can destabilize adversarial training where optimal gradients shift rapidly as both networks co-adapt.

### Why DropBlock instead of standard Dropout?

In convolutional layers neighboring activations are highly correlated, so the network can reconstruct individually dropped values from their neighbors. DropBlock drops contiguous 2x2 spatial regions, forcing the network to not rely on any single local area ([Ghiasi et al., 2018](https://arxiv.org/abs/1810.12890)).

### Why so many residual blocks?

The generator uses 5 residual blocks per stage (35 total), the critic uses 4 per stage (12 total). This is deeper than typical GAN architectures. Skip connections make this trainable by providing gradient shortcuts, letting each block learn small refinements rather than full transformations.

---

## Training

### Objective

```
L_critic     = E[D(G(z))] - E[D(x)] + lambda * E[(||grad D(x_hat)||_2 - 1)^2]
L_generator  = -E[D(G(z))]
```

Where `x_hat = alpha * x + (1 - alpha) * G(z)`, `alpha ~ U(0,1)`, `lambda = 10`.
The critic is updated 3 times per generator step.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 128 |
| Batch size | 256 |
| Epochs | 1024 |
| Generator LR | 1e-4 |
| Critic LR | 2e-4 |
| Adam betas | (0.0, 0.9) |
| Gradient penalty lambda | 10 |
| Critic steps per G step | 3 |
| LR warmup | 500 steps (linear) |
| Grad clip max norm | 10.0 |

### Dataset

52k face images in `data/`, resized to 128x128 and normalized to [-1, 1]. Split 90/10 train/val.

---

## Pretrained Weights

Download [best_model.pth](https://drive.google.com/file/d/1dWb7Vyx6_LDmkA5LwljK3AjRUsPOV_-F/view?usp=sharing) and place it in `checkpoints/`.

For intermediate epoch checkpoints, contact **parma.franek@gmail.com**.

---

## Project Structure

```
code/
  model.py        Generator and Critic definitions
  train.py        Training loop, WGAN-GP loss, checkpointing
  data_prep.py    Dataset, transforms, data loaders
  utils.py        All hyperparameters
  vis_outs.py     Generate and visualize samples
samples/          Sample generated images
checkpoints/      Saved models (not in git)
data/             Training images (not in git)
docs/TRAINING.md  Detailed training guide
requirements.txt
LICENSE
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA or MPS)
- Full list in `requirements.txt`

## License

MIT. See [LICENSE](LICENSE).
