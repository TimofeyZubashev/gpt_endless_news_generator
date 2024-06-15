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

from model_parts import Block
from tokenizer import tokenizer

seq_max_len = 300
n_embd = 512
vocab_size = tokenizer.vocab_size
n_layer = 6
n_head = 8
device = ("cuda" if torch.cuda.is_available() else "cpu")


class GPT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(seq_max_len, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = x.to(device)
                
        x_batched_input_ids = x['input_ids'][:,1:]
        x_batched_attention_mask = x['attention_mask'][:,1:]
        
        # Cохранили размерности наших ответов
        B, T = x_batched_input_ids.size(0),x_batched_input_ids.size(1)
        
        base_tensor = torch.tensor([101]+[103]*(T-1), device = device).repeat(B,1)
        
        tok_emb = self.token_embedding_table(base_tensor)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        
        x = tok_emb + pos_emb  # (B,T,C)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        return logits
        
        # Calculate loss
        loss = F.cross_entropy(logits.permute(0,2,1), x_batched_input_ids)
        
        return loss

    @torch.inference_mode()
    def generate(self, starting_seq: torch.tensor):
        self.eval()
        starting_seq = starting_seq.to(device)
        B,T_start = starting_seq.shape
        
        tail_tensor = torch.tensor([103]*(seq_max_len-T_start), device = device).repeat(B,1)
        
        
        base_tensor = torch.cat((starting_seq,tail_tensor), dim = -1)
        base_tensor = base_tensor[:,:seq_max_len]
        #print(base_tensor.shape)
    
        tok_emb = self.token_embedding_table(base_tensor)
        pos_emb = self.position_embedding_table(torch.arange(seq_max_len, device=device))
        
        x = tok_emb + pos_emb 
        
        for block in self.blocks:
            x = block(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.inference_mode()
    def generate_v2(self, starting_seq: torch.tensor, n_generate: int):
        self.eval()
        starting_seq = starting_seq.to(device)
        generated_seq = starting_seq
        
        for generation_iteration in range(n_generate):
            B,T = generated_seq.shape
            
            tok_emb = self.token_embedding_table(generated_seq)  # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
            x = tok_emb + pos_emb 
            
            for block in self.blocks:
                x = block(x) 
            
            x = self.ln_f(x)
            logits = self.lm_head(x)
            
            # taking all batches, but only last token
            generated_token = torch.argmax(logits, dim = -1)[:,-1].view(B,-1)
            generated_seq = torch.cat((generated_seq,generated_token), dim = -1)
            
        return generated_seq
    
    @torch.inference_mode()
    def beam_search(self, starting_seq: torch.tensor, n_generate: int, beam_width: int):
        self.eval()
        starting_seq = starting_seq.to(device)
        sequences = [(starting_seq, 0)]

        for _ in range(n_generate):
            all_candidates = []
            for seq, score in sequences:
                tok_emb = self.token_embedding_table(seq)  # (B, T, C)
                pos_emb = self.position_embedding_table(torch.arange(seq.size(1), device=device))  # (T, C)
                x = tok_emb + pos_emb 

                for block in self.blocks:
                    x = block(x)

                x = self.ln_f(x)
                logits = self.lm_head(x)

                # Получаем вероятности след токена для seq
                log_probs = F.softmax(logits[:, -1, :], dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                for i in range(beam_width):
                    candidate = (torch.cat([seq, top_indices[:, i].unsqueeze(-1)], dim=-1), score + top_log_probs[:, i].item())
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]

        return sequences[0][0]
    
    @torch.inference_mode()
    def beam_generate(self, starting_seq: torch.tensor, n_generate: int, beam_width: int = 3):
        self.eval()
        return self.beam_search(starting_seq, n_generate, beam_width)

model = GPT_Model()
model.to(device)