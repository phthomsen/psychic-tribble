""" SMALL PRETRAINED TRANSFORMER """
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO if not os.getenv("DEBUG") else logging.DEBUG) # not pretty but works
logger = logging.getLogger(__name__)

def get_yaml_params(yaml_path):
  with open(yaml_path) as f:
    data = yaml.safe_load(f)
  return data

# data loading
def get_batch(data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size, n_embd, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    # compute attention scores aka affinities
    wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, T, C) -> (B, T, T)
    # logger.debug(f"weight matrix in this head has: {wei.shape}")
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    # logger.debug(f"weight matrix after masking: {wei.shape}")
    wei = F.softmax(wei, dim=1) # (B, T, T)
    # logger.debug(f"weight matrix after softmax: {wei.shape}")
    wei = self.dropout(wei)
    # perform self weighted aggregation of the values
    v = self.value(x)
    out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    # logger.debug(f"after applying the complete head: {out.shape}")

    return out

# kind of like group convolution
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
     
  def forward(self, x):
    logger.debug(f"shape before concatenation: {x.shape}")
    out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the channel dimension
    logger.debug(f"shape after concatenation: {out.shape}")
    out = self.dropout(self.proj(out)) # projection back into the residual pathway
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a nonlinearity """

  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd), # grow layer
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # shrink layer
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head, dropout, block_size):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    logger.debug(f"initial shape: {x.shape}")
    x_res = self.ln1(x)
    logger.debug(f"shape after layer norm 1: {x.shape}")
    x_res = self.sa(x_res)
    logger.debug(f"shape after attention: {x.shape}")
    x = x + x_res
    logger.debug(f"shape after residual conntection 1: {x.shape}")
    # x = x + self.sa(self.ln1(x)) # residual connections
    x_res = self.ln2(x)
    logger.debug(f"shape after layer norm 2: {x.shape}")
    x_res = self.ffwd(x_res)
    logger.debug(f"shape after forward: {x.shape}")
    x = x + x_res 
    logger.debug(f"shape after residual 2: {x.shape}")
    # x = x + self.ffwd(self.ln1(x)) # residual connections
    return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device, dropout):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def main():

  device = 'cuda' if torch.cuda.is_available() else 'mps'
  torch.manual_seed(1337)
  
  store_model = True if os.getenv("STORE") else None
  load_model = True if os.getenv("LOAD") else None
     
  # load hyperparams
  yaml_params = get_yaml_params("hyper_mac.yaml")
  n_embd = yaml_params['n_embd']
  max_iters = yaml_params['max_iters']
  block_size = yaml_params['block_size']
  batch_size = yaml_params['batch_size']
  eval_iters = yaml_params['eval_iters']
  eval_interval = yaml_params['eval_interval']
  n_head = yaml_params['n_head']
  n_layer = yaml_params['n_layer']
  dropout = yaml_params['dropout']
  i_lr = float(yaml_params['learning_rate'])
  model_path = "../models/spt.pt"

  logger.info(f" --- Number of Epochs               : {max_iters}")
  logger.info(f" --- Number of layers               : {n_layer}")
  logger.info(f" --- Size of Embedding              : {n_embd}")
  logger.info(f" --- Number of heads                : {n_head}")
  logger.info(f" --- Size of Heads                  : {n_embd // n_head}")
  logger.info(f" --- Initial learning rate          : {i_lr}")
  logger.info(f" --- Number of layers               : {n_layer}") 

  # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  with open(yaml_params['data_path'], 'r', encoding='utf-8') as f:
      text = f.read()

  # here are all the unique characters that occur in this text
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

  # Train and test splits
  data = torch.tensor(encode(text), dtype=torch.long)
  n = int(0.9*len(data)) # first 90% will be train, rest val
  train_data = data[:n]
  val_data = data[n:]

  if not load_model:
    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, device, dropout)
  else:
    model = torch.load("../models/spt.pt")
  m = model.to(device)

  # number of parameters
  n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
  logger.info(f" --- Number of trainable parameters : {n_params}")
  # create a PyTorch optimizer and scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=i_lr)
  scheduler = ReduceLROnPlateau(optimizer, 'min', )

  for iter in tqdm(range(max_iters)):

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0:
          losses = estimate_loss(model, val_data, eval_iters, block_size, batch_size, device)
          logger.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
          scheduler.step(losses['val'])

      # sample a batch of data
      xb, yb = get_batch(train_data, block_size, batch_size, device)

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

  logger.info("Training done.")
  if store_model:
    os.makedirs("../models/", exist_ok=True)
    if os.path.isfile(model_path):
      if input("want to override the existing model file ? y/n \n") == "y":
        logger.info("Deleting the old model")
        os.remove(model_path)
        logger.info("Store model.")
        torch.save(model, model_path)
    else:
      logger.info("Store model")
      torch.save(model, model_path)
  
  logger.info("Start generating.")
  # generate from the model
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
  main()
