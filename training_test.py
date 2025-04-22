import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import random
from tqdm import tqdm

# --- Configuration (Minimal & Stable) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Synthetic Data Parameters ---
VOCAB_SIZE = 50  # Size of our simple vocabulary (e.g., numbers 0-49)
MAX_SEQ_LEN = 64 # Keep sequences short
DATASET_SIZE = 512 # Number of synthetic sequences
BATCH_SIZE = 16    # Smaller batch size is often more stable initially

# --- Model Hyperparameters (Minimal) ---
NUM_LAYERS = 2      # Start with very few layers
EMB_SIZE = 128      # Smaller embedding size
NHEAD = 4           # Fewer heads (must divide EMB_SIZE)
FFN_HID_DIM = 256   # Smaller feed-forward hidden dim
DROPOUT = 0.1       # Standard dropout

# --- Training Hyperparameters (Stability Focus) ---
LR = 1e-5           # Start low, even with warmup
NUM_EPOCHS = 10     # Train only for a few epochs to check stability
CLIP_GRAD = 0.5     # Keep gradient clipping low
WARMUP_STEPS = 100  # Number of steps for linear warmup (adjust based on dataset size)

# --- Special Tokens (Synthetic Data) ---
PAD_TOKEN_IDX = 0
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2
# Other tokens will be 3 to VOCAB_SIZE-1

# --- Synthetic Dataset ---
class SyntheticSequenceDataset(Dataset):
    def __init__(self, vocab_size, seq_len, dataset_size):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        self.pad_idx = PAD_TOKEN_IDX
        self.sos_idx = SOS_TOKEN_IDX
        self.eos_idx = EOS_TOKEN_IDX
        # Start sequences after special tokens
        self.data_token_start_idx = 3

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Generate a random sequence of numbers (e.g., increasing sequence)
        seq_core_len = random.randint(5, self.seq_len - 3) # Random length for core data
        start_num = random.randint(self.data_token_start_idx, self.vocab_size // 2)
        # Simple increasing sequence
        seq_core = list(range(start_num, min(self.vocab_size, start_num + seq_core_len)))

        # Add SOS and EOS
        full_sequence = [self.sos_idx] + seq_core + [self.eos_idx]

        # Pad sequence
        padding_len = self.seq_len - len(full_sequence)
        if padding_len < 0: # Truncate if somehow too long
             full_sequence = full_sequence[:self.seq_len-1] + [self.eos_idx]
             padding_len = 0
        full_sequence += [self.pad_idx] * padding_len

        # Target is shifted sequence
        input_seq = torch.tensor(full_sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(full_sequence[1:], dtype=torch.long)

        return input_seq, target_seq

# --- Positional Encoding (Standard) ---
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0) # Add batch dimension
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        seq_len = token_embedding.size(1)
        pos_emb = self.pos_embedding.narrow(1, 0, seq_len)
        return self.dropout(token_embedding + pos_emb)

# --- Transformer Model (Decoder-Only, norm_first=True) ---
class MinimalTransformerDecoder(nn.Module):
    def __init__(self, num_tokens, emb_size, nhead, ffn_hid_dim, num_layers, dropout, max_seq_len):
        super(MinimalTransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout, max_seq_len)

        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead,
            dim_feedforward=ffn_hid_dim, dropout=dropout,
            batch_first=True,
            norm_first=True # Crucial for stability
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(emb_size, num_tokens)

    def forward(self, src, tgt_mask=None, src_padding_mask=None):
        # Check input indices validity (optional but good practice)
        if torch.any(src >= self.token_embedding.num_embeddings) or torch.any(src < 0):
            raise IndexError(f"Input indices out of bounds! Max index: {src.max()}, Min index: {src.min()}, Vocab Size: {self.token_embedding.num_embeddings}")

        src_emb = self.positional_encoding(self.token_embedding(src))

        # Check for NaNs after embedding/PE
        if torch.isnan(src_emb).any(): raise RuntimeError("NaN detected after Embedding/PE")

        output = self.transformer_decoder(
            tgt=src_emb, memory=src_emb,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=src_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        # Check for NaNs after Transformer layers
        if torch.isnan(output).any(): raise RuntimeError("NaN detected after TransformerDecoder layers")

        logits = self.output_linear(output)

        # Check for NaNs in final logits
        if torch.isnan(logits).any(): raise RuntimeError("NaN detected in final logits")

        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- Weight Initialization ---
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Often initialized with normal distribution
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# --- Main Execution ---
if __name__ == "__main__":

    # --- 1. Data ---
    print("Creating synthetic dataset and dataloader...")
    train_dataset = SyntheticSequenceDataset(VOCAB_SIZE, MAX_SEQ_LEN, DATASET_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset size: {len(train_dataset)}, Dataloader steps per epoch: {len(train_dataloader)}")

    # --- 2. Model ---
    print("Initializing model...")
    model = MinimalTransformerDecoder(
        num_tokens=VOCAB_SIZE,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        ffn_hid_dim=FFN_HID_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    model.apply(init_weights) # Apply initialization
    print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 3. Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=LR, eps=1e-7) # Use base LR, scheduler will adjust. Added small eps.

    # Scheduler: Linear warmup
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step + 1) / float(max(1.0, WARMUP_STEPS))
        return 1.0 # Maintain base LR after warmup
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Warming up for {WARMUP_STEPS} steps.")

    # --- 4. Training Loop ---
    print("Starting Training...")
    # torch.autograd.set_detect_anomaly(True) # Enable if NaNs reappear

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch}/{NUM_EPOCHS}", leave=False)

        for i, (src, tgt) in enumerate(batch_iterator):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            seq_len = src.size(1) # Should be MAX_SEQ_LEN - 1
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
            src_padding_mask = (src == PAD_TOKEN_IDX) # Boolean mask

            # Verify mask shapes and types (optional, for debugging)
            # if i == 0 and epoch == 1:
            #     print(f"tgt_mask shape: {tgt_mask.shape}, dtype: {tgt_mask.dtype}")
            #     print(f"src_padding_mask shape: {src_padding_mask.shape}, dtype: {src_padding_mask.dtype}")

            optimizer.zero_grad()

            try:
                logits = model(src, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)

                # Check logits just before loss
                if torch.isnan(logits).any(): raise RuntimeError(f"NaN detected in logits before loss calculation!")

                loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))

                if torch.isnan(loss): raise RuntimeError(f"NaN detected in loss!")

                loss.backward() # Compute gradients

                # Check gradients before clipping (optional, for debugging)
                # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                # print(f"DEBUG: Grad Norm BEFORE clipping (Batch {i}): {total_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD) # Clip gradients
                optimizer.step() # Update weights
                scheduler.step() # Update learning rate

                total_loss += loss.item()
                current_lr = scheduler.get_last_lr()[0]
                batch_iterator.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

            except Exception as e:
                 print(f"\nERROR during training step {i} in epoch {epoch}: {e}")
                 # Print details about the batch that failed
                 print("Failed batch src sample:", src[0,:20])
                 print("Failed batch tgt sample:", tgt[0,:20])
                 raise e # Reraise the error to stop training


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} Summary: Average Training Loss: {avg_loss:.4f}, Final LR: {scheduler.get_last_lr()[0]:.8f}")

    print("Minimal Training Finished Successfully!")