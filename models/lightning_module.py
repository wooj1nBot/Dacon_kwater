import pytorch_lightning as pl
import torch.optim.optimizer
from typing import Tuple
from torch.optim import AdamW
import torch.nn.functional as F
import wandb
from models.Haea.utils.configs import ARTrainConfig, FinetuningConfig, DVAETrainConfig
from models.Haea.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from typing import List
from models.Haea.model.swin_unet import VVRSNet
from models.Haea.model.model import Haea
from models.Haea.modules.swin import SwinDiscreteVAE