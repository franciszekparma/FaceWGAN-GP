import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dropblock import DropBlock2D

from utils import (
  LATENT_DIM, G_MAX_DIM, G_OUTPUT_DIM, G_LAYER_REPEAT, G_SKIP_LAST_LAYERS,
  D_NUM_CONV_LAYERS, D_NUM_MLP_LAYERS, D_LAYER_REPEAT_CONV,
  D_LAYER_REPEAT_LAST_CONV, D_LAYER_REPEAT_MLP, D_IN_CHANNELS,
  D_OUT_FEATURES, D_INIT_IN_CHANNELS, D_DROP_PROB, D_BLOCK_SIZE,
  D_MLP_FIRST_LAYER_DROPOUT, D_MLP_DROPOUT, IMAGE_SHAPE,
)



class GeneratorLayer(nn.Module):
  def __init__(self, in_channels, out_channels, layer_repeat, last_layer=False):
    super().__init__()

    if not last_layer:
      self.conv_block = nn.Sequential(
        nn.ConvTranspose2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=2,
          stride=2,
          padding=0,
          bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
      )
    else:
      self.conv_block = nn.Sequential(
        nn.ConvTranspose2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=2,
          stride=2,
          padding=0,
          bias=True
        )
      )

    self.residual_blocks = nn.ModuleList()

    if layer_repeat >= 1 and not last_layer:
      for _ in range(layer_repeat):
        self.residual_blocks.append(nn.Sequential(
          nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.GELU()
        ))
    elif layer_repeat >= 1 and last_layer:
      for _ in range(layer_repeat):
        self.residual_blocks.append(nn.Sequential(
          nn.BatchNorm2d(out_channels),
          nn.GELU(),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        ))

  def forward(self, X):
    x = self.conv_block(X)
    for block in self.residual_blocks:
      x = x + block(x)
    return x


class Generator(nn.Module):
  def __init__(self, device, latent_dim=LATENT_DIM, max_dim=G_MAX_DIM, output_dim=G_OUTPUT_DIM, layer_repeat=G_LAYER_REPEAT, skip_last_layers=G_SKIP_LAST_LAYERS, target_image_shape=IMAGE_SHAPE):
    super().__init__()

    self.device = device
    self.latent_dim = latent_dim

    num_iter = int(math.log2(target_image_shape))
    dims = [latent_dim]

    i = 1
    while max(dims) != max_dim:
      dims.append(dims[i - 1] * 2)
      i += 1

    for n in range(1, num_iter + 1):
      prev_dim = dims[len(dims) - 1]

      if n != num_iter:
        dims.append(prev_dim // 2)

      else:
        dims.append(output_dim)


    self.generator = nn.Sequential()

    prev_dim = -1

    for i in range(len(dims)):
      if i < len(dims) - 1 - skip_last_layers - 1:
        prev_dim = dims[i]
        self.generator.append(
          GeneratorLayer(
            in_channels=dims[i],
            out_channels=dims[i + 1],
            layer_repeat=layer_repeat
          )
          )
      elif i == len(dims) - 2:
        self.generator.append(
          GeneratorLayer(
            in_channels=prev_dim//2,
            out_channels=dims[i + 1],
            layer_repeat=layer_repeat,
            last_layer=True
          )
        )

  def forward(self, batch_size):
    latent = torch.randn((batch_size, self.latent_dim), device=self.device).reshape(-1, self.latent_dim, 1, 1)
    return torch.tanh(self.generator(latent))




class DiscriminatorConvLayer(nn.Module):
  def __init__(self, in_channels, out_channels, layer_repeat, drop_prob=D_DROP_PROB, block_size=D_BLOCK_SIZE, last_layer=False):
    super().__init__()

    if not last_layer:
      self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.GELU(),
        DropBlock2D(block_size=block_size, drop_prob=drop_prob),
      )
    else:
      self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
      )

    self.residual_blocks = nn.ModuleList()
    self.last_layer = last_layer

    if layer_repeat >= 1 and not last_layer:
      for _ in range(layer_repeat):
        self.residual_blocks.append(nn.Sequential(
          nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
          nn.InstanceNorm2d(out_channels, affine=True),
          nn.GELU()
        ))
    elif layer_repeat >= 1 and last_layer:
      for _ in range(layer_repeat):
        self.residual_blocks.append(nn.Sequential(
          nn.InstanceNorm2d(out_channels, affine=True),
          nn.GELU(),
          nn.Conv2d(out_channels, out_channels, 4, 2, 1, bias=True)
        ))

  def forward(self, X):
    x = self.conv_block(X)
    for block in self.residual_blocks:
      if not self.last_layer:
        x = x + block(x)
      else:
        x = block(x)
    return x


class DiscriminatorMLPLayer(nn.Module):
  def __init__(self, in_features, out_features, first_layer_dropout=D_MLP_FIRST_LAYER_DROPOUT, dropout=D_MLP_DROPOUT, layer_repeat=0, first_layer=False, last_layer=False):
    super().__init__()

    if not last_layer and not first_layer:
      self.mlp_block = nn.Sequential(
        nn.Linear(
          in_features=in_features,
          out_features=out_features,
          bias=True
        ),
        nn.GELU(),
        nn.Dropout(dropout)
    )

    elif last_layer:
      self.mlp_block = nn.Sequential(
        nn.Linear(
          in_features=in_features,
          out_features=out_features,
          bias=True
        )
    )

    else:
      self.mlp_block = nn.Sequential(
        nn.Linear(
          in_features=in_features,
          out_features=out_features,
          bias=True
        ),
        nn.GELU(),
        nn.Dropout(first_layer_dropout)
    )

    if layer_repeat >= 1 and not last_layer:
      for _ in range(layer_repeat):
        self.mlp_block.append(nn.Sequential(
          nn.Linear(out_features, out_features, bias=True),
          nn.GELU(),
          nn.Dropout(dropout)
        ))
    elif layer_repeat >= 1 and last_layer:
      for _ in range(layer_repeat):
        self.mlp_block.append(nn.Sequential(
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(out_features, out_features, bias=True)
        ))

  def forward(self, X):
    return self.mlp_block(X)


class Discriminator(nn.Module):
  def __init__(self, num_conv_layers=D_NUM_CONV_LAYERS, num_mlp_layers=D_NUM_MLP_LAYERS, layer_repeat_conv=D_LAYER_REPEAT_CONV, layer_repeat_last_conv=D_LAYER_REPEAT_LAST_CONV, layer_repeat_mlp=D_LAYER_REPEAT_MLP, in_channels=D_IN_CHANNELS, out_features=D_OUT_FEATURES, init_in_channels=D_INIT_IN_CHANNELS, image_shape=IMAGE_SHAPE):
    super().__init__()

    self.discriminator = nn.Sequential()

    initial_pow = -1

    i = 0
    while True:
      if 2**i > in_channels and 2**i >= init_in_channels:
        initial_pow = i
        break

      i += 1

    pows = [initial_pow + i for i in range(num_conv_layers)]

    for n in range(num_conv_layers):
      if n != num_conv_layers - 1 and n != 0:
        self.discriminator.append(DiscriminatorConvLayer(
          in_channels=2**pows[n - 1],
          out_channels=2**pows[n],
          layer_repeat=layer_repeat_conv
        ))

      elif n == 0:
        self.discriminator.append(DiscriminatorConvLayer(
          in_channels=in_channels,
          out_channels=2**pows[n],
          layer_repeat=layer_repeat_conv
        ))

      else:
        self.discriminator.append(DiscriminatorConvLayer(
          in_channels=2**pows[n - 1],
          out_channels=2**pows[n],
          layer_repeat=layer_repeat_last_conv,
          last_layer=True
        ))


    self.discriminator.append(nn.Flatten())

    with torch.no_grad():
      dummy = torch.zeros(1, in_channels, image_shape, image_shape)
      for layer in self.discriminator:
          dummy = layer(dummy)

    out_img_size = dummy.shape[1]

    layers_in = [out_img_size]

    for n in range(num_mlp_layers - 1):
      layers_in.append(layers_in[-1] // 2)

    for n in range(num_mlp_layers):
      if n != num_mlp_layers - 1 and n != 0:
        self.discriminator.append(DiscriminatorMLPLayer(
          in_features=layers_in[n],
          out_features=layers_in[n + 1],
          layer_repeat=layer_repeat_mlp,
        ))
      elif n == num_mlp_layers - 1:
        self.discriminator.append(DiscriminatorMLPLayer(
          in_features=layers_in[n],
          out_features=out_features,
          last_layer=True
        ))
      else:
        self.discriminator.append(DiscriminatorMLPLayer(
          in_features=layers_in[n],
          out_features=layers_in[n + 1],
          layer_repeat=layer_repeat_mlp,
          first_layer=True
        ))

  def forward(self, X):
    return self.discriminator(X)
