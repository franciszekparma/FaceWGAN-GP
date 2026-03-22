import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -- Device & Reproducibility ----------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
SEED = 24

# -- Training --------------------------------------------
EPOCHS = 1024
N_CRITIC = 3
WARMUP_STEPS = 500
LR_G = 1e-4
LR_D = 2e-4
BETAS = (0.0, 0.9)
WEIGHT_DECAY = 0.0
GRAD_CLIP_MAX_NORM = 10.0
GP_LAMBDA = 10

# -- Checkpointing ---------------------------------------
SAVE_EVERY = 5
LOAD_STATE = False
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# -- Generator -------------------------------------------
LATENT_DIM = 128
G_MAX_DIM = 512
G_OUTPUT_DIM = 3
G_LAYER_REPEAT = 5
G_SKIP_LAST_LAYERS = 2

# -- Discriminator (conv) ---------------------------------
D_NUM_CONV_LAYERS = 4
D_NUM_MLP_LAYERS = 4
D_LAYER_REPEAT_CONV = 4
D_LAYER_REPEAT_LAST_CONV = 3
D_LAYER_REPEAT_MLP = 0
D_IN_CHANNELS = 3
D_OUT_FEATURES = 1
D_INIT_IN_CHANNELS = 64
D_DROP_PROB = 0.1
D_BLOCK_SIZE = 2

# -- Discriminator (MLP) -----------------------------------
D_MLP_FIRST_LAYER_DROPOUT = 0.2
D_MLP_DROPOUT = 0.1

# -- Data --------------------------------------------------
IMAGE_SHAPE = 128
BATCH_SIZE = 256
NUM_WORKERS = 12
TRAIN_SPLIT = 0.9
DATA_DIR = str(PROJECT_ROOT / "data")

# -- Visualization -----------------------------------------
SHOW_BATCH_SIZE = 32

# -- Weight init -------------------------------------------
INIT_MEAN = 0.0
INIT_STD = 0.02
