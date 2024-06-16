# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Transformers
import transformers
from transformers import AutoTokenizer, AutoModel

# Model creation 
import torch
from torch import nn
import torch.nn.functional as F

# Dataset & Dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Optimizer 
from torch.optim import Adam

# Schedular
from torch.optim.lr_scheduler import ReduceLROnPlateau

# TQDM
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
from IPython.display import clear_output

from model import model
from tokenizer import tokenizer

bs = 2
starting_tensor = torch.tensor([101]).repeat(bs,1)

for text in model.beam_generate(starting_tensor,5):
    print(tokenizer.decode(text.tolist()))
    print()