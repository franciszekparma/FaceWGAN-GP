import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model import Generator, Discriminator
from data_prep import train_dl, val_dl
from utils import (
  DEVICE, SEED, EPOCHS, N_CRITIC, WARMUP_STEPS, LR_G, LR_D, BETAS,
  WEIGHT_DECAY, GRAD_CLIP_MAX_NORM, GP_LAMBDA, SAVE_EVERY, LOAD_STATE,
  INIT_MEAN, INIT_STD, CHECKPOINT_DIR, BEST_MODEL_PATH,
)

torch.manual_seed(SEED)


def init_weights(module):
  if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    nn.init.normal_(module.weight, mean=INIT_MEAN, std=INIT_STD)

    if module.bias is not None:
      nn.init.zeros_(module.bias)

  elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)


def gradient_penalty(discriminator, X_real, X_fake, device=DEVICE, lam=GP_LAMBDA):
  B = X_real.shape[0]

  alpha = torch.rand((B, 1, 1, 1), device=device)

  X_hat = alpha * X_real + (1 - alpha) * X_fake.detach()
  X_hat.requires_grad_(True)

  score = discriminator(X_hat)
  grads = torch.autograd.grad(
    outputs=score, inputs=X_hat,
    grad_outputs=torch.ones_like(score),
    create_graph=True, retain_graph=True
  )[0]

  norm = grads.view(B, -1).norm(2, dim=1)

  return lam * ((norm - 1) ** 2).mean()


def validate(G, D, val_dl):
  G.eval()
  D.eval()
  d_losses = []
  g_losses = []

  with torch.no_grad():
    for n_batch, imgs in enumerate(val_dl):
      real_imgs = imgs.to(DEVICE)
      batch_size = imgs.shape[0]

      gen_imgs = G(batch_size)

      score_fake = D(gen_imgs)
      score_real = D(real_imgs)

      d_loss = torch.mean(score_fake) - torch.mean(score_real)
      g_loss = -torch.mean(score_fake)

      d_losses.append(d_loss.item())
      g_losses.append(g_loss.item())

  return sum(d_losses) / len(d_losses), sum(g_losses) / len(g_losses)


def lr_lambda(step):
  if step < WARMUP_STEPS:
    return step / WARMUP_STEPS
  return 1.0



def main():
  G = Generator(DEVICE).to(DEVICE)
  D = Discriminator().to(DEVICE)

  G.apply(init_weights)
  D.apply(init_weights)

  optim_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS, weight_decay=WEIGHT_DECAY)
  optim_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS, weight_decay=WEIGHT_DECAY)

  sched_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda)
  sched_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lr_lambda)


  best_wasserstein = float('inf')


  CHECKPOINT_DIR.mkdir(exist_ok=True)

  if LOAD_STATE:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])
    optim_G.load_state_dict(checkpoint['optim_G'])
    optim_D.load_state_dict(checkpoint['optim_D'])

    val_g_loss = checkpoint['val_g_loss']
    val_d_loss = checkpoint['val_d_loss']
    best_wasserstein = checkpoint['best_wasserstein']


  for epoch in tqdm(range(EPOCHS)):
    G.train()
    D.train()

    d_losses = []
    g_losses = []

    for n_batch, imgs in enumerate(train_dl):
      real_imgs = imgs.to(DEVICE)
      batch_size = imgs.shape[0]

      for _ in range(N_CRITIC):
        gen_imgs = G(batch_size)

        d_loss = torch.mean(D(gen_imgs.detach())) - torch.mean(D(real_imgs)) + gradient_penalty(D, real_imgs, gen_imgs, DEVICE)

        optim_D.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optim_D.step()

      sched_D.step()

      d_losses.append(d_loss.item())

      gen_imgs = G(batch_size)
      g_loss = -torch.mean(D(gen_imgs))
      g_losses.append(g_loss.item())

      optim_G.zero_grad()
      g_loss.backward()
      torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
      optim_G.step()
      sched_G.step()


    val_d_loss, val_g_loss = validate(G, D, val_dl)

    print(f"--------------------------------------------------")
    print(f"Epoch: {epoch}")
    print("Train:")
    print(f"  D Loss: {sum(d_losses)/len(d_losses):.4f} | G Loss: {sum(g_losses)/len(g_losses):.4f}")
    print("Val:")
    print(f"  D Loss: {val_d_loss:.4f} | G Loss: {val_g_loss:.4f}")
    print(f"--------------------------------------------------\n")



    if epoch % SAVE_EVERY == 0:
      torch.save({
          'G': G.state_dict(),
          'D': D.state_dict(),
          'optim_G': optim_G.state_dict(),
          'optim_D': optim_D.state_dict(),
          'epoch': epoch,
          'val_g_loss': val_g_loss,
          'val_d_loss': val_d_loss,
          'best_wasserstein': best_wasserstein
        }, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
        )
      print(f"Periodic save at epoch {epoch}\n")

    if val_d_loss < best_wasserstein:
      best_wasserstein = val_d_loss
      torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'optim_G': optim_G.state_dict(),
        'optim_D': optim_D.state_dict(),
        'epoch': epoch,
        'val_g_loss': val_g_loss,
        'val_d_loss': val_d_loss,
        'best_wasserstein': best_wasserstein
      }, BEST_MODEL_PATH
      )
      print(f"!!!Saved best model (Wasserstein: {val_d_loss:.4f})!!!\n")


if __name__ == "__main__":
  main()
