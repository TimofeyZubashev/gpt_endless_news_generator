from tokenizer import tokenizer

seq_max_len = 300
n_embd = 512
vocab_size = tokenizer.vocab_size
n_layer = 6
n_head = 8
device = ("cuda" if torch.cuda.is_available() else "cpu")

model_params = {}