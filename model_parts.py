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

from tokenizer import tokenizer


seq_max_len = 300
block_size = seq_max_len
n_embd = 512
vocab_size = tokenizer.vocab_size
n_layer = 6
n_head = 8
device = ("cuda" if torch.cuda.is_available() else "cpu")

class Head_Masked(nn.Module):
    """One head of self-attention with masking."""
    def __init__(self, head_size,dropout_prob=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))  # (B, T, T)

        # Apply mask
        #mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, 1, T) * (B, T, 1) -> (B, T, T)
        
        scores = scores.masked_fill(self.tril[:T,:T] == 0, float(-1e9))  # Apply lower triangular mask
        #scores = scores.masked_fill(mask[:T,:T] == 0, float(-1e9))  # Apply attention mask
        
        # Normalize the scores
        attn = F.softmax(scores, dim=-1)  # (B, T, T)
        attn = self.dropout(attn)

        # Apply attention weights to values
        out = attn @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head_Masked(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p = 0.2),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size почему то деление n_embd на кол-во голов attention
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
