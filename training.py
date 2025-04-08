import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pretty_midi
import numpy as np
import os
import glob
import math
import pickle
from collections import Counter
from tqdm import tqdm # For progress bars

# --- Constants and Configuration ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Tokenization Parameters ---
# --- YOU MUST CAREFULLY DEFINE AND REFINE THESE ---
TIME_RESOLUTION = 0.02  # seconds per time step token (e.g., 50 steps per second)
MAX_TIME_SHIFT = 2.0    # Max time shift to represent directly (seconds)
VELOCITY_BINS = 4       # Number of bins for velocity (e.g., 0-31, 32-63, 64-95, 96-127)
MAX_VOICES = 10         # Maximum number of harmony voices supported + 1 (for token)

# --- Special Tokens ---
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>" # Start of Sequence
EOS_TOKEN = "<eos>" # End of Sequence
SEP_TOKEN = "<sep>" # Separator between melody and harmony
UNKNOWN_TOKEN = "<unk>"

# --- Model Hyperparameters (EXAMPLES - TUNE THESE!) ---
VOCAB_SIZE = -1 # Will be determined after building vocabulary
EMB_SIZE = 512
NHEAD = 8
NUM_LAYERS = 6
FFN_HID_DIM = 2048
DROPOUT = 0.1
MAX_SEQ_LEN = 1024 # Max sequence length the model can handle

# --- Training Hyperparameters (EXAMPLES - TUNE THESE!) ---
LR = 0.0001
BATCH_SIZE = 16 # Adjust based on GPU memory
NUM_EPOCHS = 50
CLIP_GRAD = 1.0 # Gradient clipping value

# --- Data Paths ---
DATASET_ROOT = 'path/to/your/dataset_root/' # <<< CHANGE THIS
TOKENIZER_PATH = 'tokenizer.pkl'
MODEL_SAVE_PATH = 'music_transformer_model.pth'

# --- Helper Functions ---

def velocity_to_bin(velocity, bins=VELOCITY_BINS):
    """Maps MIDI velocity (0-127) to a discrete bin."""
    if velocity == 0: return 0 # Note off case sometimes uses velocity 0
    return min(bins - 1, int(velocity / (128.0 / bins)))

def time_to_bin(time_delta, resolution=TIME_RESOLUTION, max_time=MAX_TIME_SHIFT):
    """Maps a time delta (seconds) to a discrete time shift token bin."""
    if time_delta >= max_time:
        return int(max_time / resolution) -1
    elif time_delta < 0: # Should not happen with proper sorting
        return 0
    else:
        return int(time_delta / resolution)

def build_vocabulary(dataset_root, time_res, max_time, vel_bins, max_voices):
    """Scans dataset to build token vocabulary."""
    print("Building vocabulary...")
    token_counts = Counter()
    all_song_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
                     if os.path.isdir(os.path.join(dataset_root, d))]

    num_time_bins = int(max_time / time_res)

    for song_dir in tqdm(all_song_dirs):
        melody_files = glob.glob(os.path.join(song_dir, 'melody*.mid'))
        harmony_files = [f for f in glob.glob(os.path.join(song_dir, 'harmony*.mid'))
                         if f not in melody_files]

        if not melody_files: continue
        melody_file = melody_files[0]

        try:
            # --- Process Melody ---
            midi_data = pretty_midi.PrettyMIDI(melody_file)
            instrument = midi_data.instruments[0] # Assume melody is first instrument
            instrument.notes.sort(key=lambda x: x.start)
            last_event_time = 0.0
            for note in instrument.notes:
                delta_t = note.start - last_event_time
                token_counts[f"TIME_{time_to_bin(delta_t, time_res, max_time)}"] += 1
                token_counts[f"NOTEON_{note.pitch}"] += 1
                token_counts[f"VEL_{velocity_to_bin(note.velocity, vel_bins)}"] += 1
                # Represent duration implicitly via next TIME shift or NOTE_OFF
                # Or explicitly add DURATION tokens (more complex)
                last_event_time = note.start # Use start time for simplicity

            # --- Process Harmony (Collect all harmony notes first) ---
            all_harmony_notes = []
            for hf in harmony_files:
                 midi_h = pretty_midi.PrettyMIDI(hf)
                 for inst in midi_h.instruments:
                     all_harmony_notes.extend(inst.notes)

            all_harmony_notes.sort(key=lambda x: x.start)
            last_event_time = 0.0 # Reset for harmony part
            for note in all_harmony_notes:
                 delta_t = note.start - last_event_time
                 token_counts[f"TIME_{time_to_bin(delta_t, time_res, max_time)}"] += 1
                 token_counts[f"NOTEON_{note.pitch}"] += 1
                 token_counts[f"VEL_{velocity_to_bin(note.velocity, vel_bins)}"] += 1
                 last_event_time = note.start

        except Exception as e:
            print(f"Skipping {song_dir} due to error: {e}")
            continue

    # --- Create Mappings ---
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNKNOWN_TOKEN]
    vocab.extend([f"VOICE_{i}" for i in range(1, max_voices + 1)])
    vocab.extend([f"NOTEON_{p}" for p in range(128)])
    vocab.extend([f"VEL_{v}" for v in range(vel_bins)])
    vocab.extend([f"TIME_{t}" for t in range(num_time_bins)])
    # Add NOTE_OFF tokens if needed, or Duration tokens

    # Keep only tokens seen in the data (optional, reduces vocab size)
    # min_freq = 2
    # final_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNKNOWN_TOKEN]
    # final_vocab.extend([t for t, c in token_counts.items() if c >= min_freq])
    # final_vocab.extend([f"VOICE_{i}" for i in range(1, max_voices + 1)]) # Ensure voice tokens are there

    # Use all possible systematic tokens for robustness
    final_vocab = list(dict.fromkeys(vocab)) # Remove duplicates if any systematic overlap

    token_to_id = {token: i for i, token in enumerate(final_vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    print(f"Vocabulary size: {len(final_vocab)}")
    return token_to_id, id_to_token

def save_tokenizer(token_to_id, id_to_token, path):
    with open(path, 'wb') as f:
        pickle.dump({'token_to_id': token_to_id, 'id_to_token': id_to_token}, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['token_to_id'], data['id_to_token']

def midi_to_tokens(midi_files, token_to_id, time_res, max_time, vel_bins):
    """
    Converts a list of MIDI file paths into a sequence of token IDs.
    This function needs significant refinement for handling multiple harmony files correctly.
    Current version processes them sequentially based on time.
    """
    tokens = []
    all_notes = []
    is_melody = True # Flag to distinguish? Or rely on SEP token?

    # --- Collect all notes with timestamps ---
    # Assign a track index or rely on temporal ordering
    for i, file_path in enumerate(midi_files):
         try:
             midi_data = pretty_midi.PrettyMIDI(file_path)
             for instrument in midi_data.instruments:
                 for note in instrument.notes:
                     # Store start time, pitch, velocity, maybe original track index i
                     all_notes.append({'start': note.start, 'pitch': note.pitch, 'velocity': note.velocity, 'track': i})
         except Exception as e:
             print(f"Warning: Could not parse {file_path}: {e}")
             continue # Skip corrupted file

    # --- Sort all notes globally by start time ---
    all_notes.sort(key=lambda x: x['start'])

    # --- Convert sorted notes to tokens ---
    last_event_time = 0.0
    for note in all_notes:
        delta_t = note['start'] - last_event_time
        time_token = f"TIME_{time_to_bin(delta_t, time_res, max_time)}"
        note_token = f"NOTEON_{note['pitch']}"
        vel_token = f"VEL_{velocity_to_bin(note['velocity'], vel_bins)}"

        tokens.append(token_to_id.get(time_token, token_to_id[UNKNOWN_TOKEN]))
        tokens.append(token_to_id.get(note_token, token_to_id[UNKNOWN_TOKEN]))
        tokens.append(token_to_id.get(vel_token, token_to_id[UNKNOWN_TOKEN]))
        # Add NOTE_OFF or DURATION tokens here if using that strategy

        last_event_time = note['start']

    return tokens


# --- Dataset Class ---

class MIDIDataset(Dataset):
    def __init__(self, root_dir, token_to_id, max_seq_len, max_voices, time_res, max_time, vel_bins):
        self.root_dir = root_dir
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.max_seq_len = max_seq_len
        self.max_voices = max_voices
        self.time_res = time_res
        self.max_time = max_time
        self.vel_bins = vel_bins
        self.pad_id = token_to_id[PAD_TOKEN]
        self.sos_id = token_to_id[SOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.sep_id = token_to_id[SEP_TOKEN]

        self.song_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]
        # Filter out songs with no melody or too many voices early?

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, idx):
        song_dir = self.song_dirs[idx]
        melody_files = glob.glob(os.path.join(song_dir, 'melody*.mid'))
        harmony_files = sorted([f for f in glob.glob(os.path.join(song_dir, 'harmony*.mid'))
                         if f not in melody_files]) # Sort for consistency

        if not melody_files:
             # Should ideally filter these out beforehand or return None and handle in collate_fn
             print(f"Warning: No melody found in {song_dir}")
             # Return dummy data or raise error; Returning dummy for now
             dummy_seq = [self.pad_id] * self.max_seq_len
             return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        melody_file = melody_files[0]
        num_harmony_voices = len(harmony_files)

        if num_harmony_voices == 0:
            print(f"Warning: No harmony files found in {song_dir}")
            # Return dummy data or raise error; Returning dummy for now
            dummy_seq = [self.pad_id] * self.max_seq_len
            return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)


        if num_harmony_voices > self.max_voices:
            print(f"Warning: Too many voices ({num_harmony_voices}) in {song_dir}, skipping.")
            # Return dummy data or raise error; Returning dummy for now
            dummy_seq = [self.pad_id] * self.max_seq_len
            return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)


        voice_token_str = f"VOICE_{num_harmony_voices}"
        voice_token_id = self.token_to_id.get(voice_token_str, self.token_to_id[UNKNOWN_TOKEN])

        # --- Tokenize ---
        # Pass only the single melody file
        melody_tokens = midi_to_tokens([melody_file], self.token_to_id, self.time_res, self.max_time, self.vel_bins)
        # Pass the list of harmony files
        harmony_tokens = midi_to_tokens(harmony_files, self.token_to_id, self.time_res, self.max_time, self.vel_bins)

        # --- Combine and Pad/Truncate ---
        # Format: SOS + VOICE + MELODY + SEP + HARMONY + EOS
        input_sequence = [self.sos_id] + [voice_token_id] + melody_tokens + [self.sep_id] + harmony_tokens + [self.eos_id]

        # Truncate if too long
        if len(input_sequence) > self.max_seq_len:
            input_sequence = input_sequence[:self.max_seq_len -1] + [self.eos_id] # Ensure EOS is last token if truncated

        # Target sequence is shifted input
        target_sequence = input_sequence[1:]
        input_sequence = input_sequence[:-1] # Input stops before the final target token

        # Pad sequences
        input_padding_len = self.max_seq_len - len(input_sequence)
        target_padding_len = self.max_seq_len - len(target_sequence)

        input_sequence += [self.pad_id] * input_padding_len
        target_sequence += [self.pad_id] * target_padding_len

        # Sanity check lengths
        assert len(input_sequence) == self.max_seq_len, f"Input length mismatch: {len(input_sequence)}"
        assert len(target_sequence) == self.max_seq_len, f"Target length mismatch: {len(target_sequence)}"


        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)

# --- Positional Encoding ---

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
        self.register_buffer('pos_embedding', pos_embedding) # Not a model parameter

    def forward(self, token_embedding: torch.Tensor):
        # token_embedding shape: [batch_size, seq_len, emb_size]
        # pos_embedding shape: [1, maxlen, emb_size]
        seq_len = token_embedding.size(1)
        # Use narrow to select the needed length, avoids resizing buffer if seq_len < maxlen
        pos_emb = self.pos_embedding.narrow(1, 0, seq_len)
        # Add positional encoding to token embedding
        return self.dropout(token_embedding + pos_emb)


# --- Transformer Model (Decoder-Only) ---

class MusicTransformerDecoder(nn.Module):
    def __init__(self, num_tokens, emb_size, nhead, ffn_hid_dim, num_layers, dropout, max_seq_len):
        super(MusicTransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout, max_seq_len)

        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=ffn_hid_dim, dropout=dropout,
                                                batch_first=True) # Important: batch_first=True
        # Use TransformerDecoder which stacks layers
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(emb_size, num_tokens)

    def forward(self, src, tgt_mask=None, src_padding_mask=None):
        # src shape: [batch_size, seq_len]
        # tgt_mask: [seq_len, seq_len] (causal mask)
        # src_padding_mask: [batch_size, seq_len] (True where padded)

        src_emb = self.positional_encoding(self.token_embedding(src))
        # src_emb shape: [batch_size, seq_len, emb_size]

        # Decoder only: use src as both target and memory input
        # The tgt_mask ensures causality.
        # memory_key_padding_mask handles padding in the "memory" (which is src itself)
        # tgt_key_padding_mask handles padding in the target sequence during self-attention
        output = self.transformer_decoder(tgt=src_emb, memory=src_emb, # Use src_emb as memory
                                          tgt_mask=tgt_mask,
                                          memory_mask=None, # No memory mask needed if memory=target
                                          tgt_key_padding_mask=src_padding_mask,
                                          memory_key_padding_mask=src_padding_mask)
        # output shape: [batch_size, seq_len, emb_size]

        logits = self.output_linear(output)
        # logits shape: [batch_size, seq_len, num_tokens]
        return logits

    def generate_square_subsequent_mask(self, sz):
        """Generates a square causal mask for the sequence length sz."""
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# --- Training Function ---

def train_epoch(model, dataloader, optimizer, criterion, pad_idx, clip_value):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        # src shape: [batch_size, seq_len]
        # tgt shape: [batch_size, seq_len]

        seq_len = src.size(1)
        tgt_mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
        # src_padding_mask shape: [batch_size, seq_len] - True where padded
        src_padding_mask = (src == pad_idx)

        optimizer.zero_grad()

        logits = model(src, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)
        # logits shape: [batch_size, seq_len, vocab_size]
        # tgt shape: [batch_size, seq_len]

        # Reshape for CrossEntropyLoss: needs [N, C] and [N]
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches

# --- Main Execution ---

if __name__ == "__main__":
    # --- 1. Build or Load Vocabulary ---
    if not os.path.exists(TOKENIZER_PATH):
        token_to_id, id_to_token = build_vocabulary(
            DATASET_ROOT, TIME_RESOLUTION, MAX_TIME_SHIFT, VELOCITY_BINS, MAX_VOICES
        )
        save_tokenizer(token_to_id, id_to_token, TOKENIZER_PATH)
    else:
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        token_to_id, id_to_token = load_tokenizer(TOKENIZER_PATH)

    VOCAB_SIZE = len(token_to_id)
    PAD_IDX = token_to_id[PAD_TOKEN]

    # --- 2. Create Datasets and Dataloaders ---
    # Consider splitting your data into train/validation sets
    # For simplicity, using the whole dataset for training here
    train_dataset = MIDIDataset(
        root_dir=DATASET_ROOT,
        token_to_id=token_to_id,
        max_seq_len=MAX_SEQ_LEN,
        max_voices=MAX_VOICES,
        time_res=TIME_RESOLUTION,
        max_time=MAX_TIME_SHIFT,
        vel_bins=VELOCITY_BINS
    )

    # Use num_workers > 0 for faster loading if not on Windows or if using Linux/macOS properly
    # pin_memory=True can speed up CPU->GPU transfer
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True) # Set num_workers based on your OS/setup


    # --- 3. Initialize Model, Loss, Optimizer ---
    model = MusicTransformerDecoder(
        num_tokens=VOCAB_SIZE,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        ffn_hid_dim=FFN_HID_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore padding in loss calculation
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) # Example scheduler

    print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 4. Training Loop ---
    print("Starting Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, train_dataloader, optimizer, criterion, PAD_IDX, CLIP_GRAD)
        # Add validation loop here if you have a validation set
        # scheduler.step() # Step the scheduler if using one

        print(f"Epoch {epoch}/{NUM_EPOCHS}, Average Training Loss: {avg_loss:.4f}")

        # --- 5. Save Model Checkpoint ---
        if epoch % 5 == 0 or epoch == NUM_EPOCHS: # Save every 5 epochs and at the end
            checkpoint_path = f"model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                # Include tokenizer info for reproducibility
                'token_to_id': token_to_id,
                'id_to_token': id_to_token,
                'config': { # Save key hyperparameters
                    'vocab_size': VOCAB_SIZE,
                    'emb_size': EMB_SIZE,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'ffn_hid_dim': FFN_HID_DIM,
                    'max_seq_len': MAX_SEQ_LEN,
                    'dropout': DROPOUT,
                    'time_res': TIME_RESOLUTION,
                    'max_time': MAX_TIME_SHIFT,
                    'vel_bins': VELOCITY_BINS,
                    'max_voices': MAX_VOICES
                }
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training Finished.")
    # --- 6. Save Final Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")