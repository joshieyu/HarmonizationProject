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
LR = 5e-6
BATCH_SIZE = 16 # Adjust based on GPU memory
NUM_EPOCHS = 50
CLIP_GRAD = 0.5 # Gradient clipping value

# --- Data Paths ---
DATASET_ROOT = './ProcessedDataset' # <<< CHANGE THIS
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
        self.max_voices = max_voices # This is the configured maximum
        self.time_res = time_res
        self.max_time = max_time
        self.vel_bins = vel_bins
        self.pad_id = token_to_id[PAD_TOKEN]
        self.sos_id = token_to_id[SOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.sep_id = token_to_id[SEP_TOKEN]

        self.song_dirs = []
        print("Scanning dataset directories...")
        for d in tqdm(os.listdir(root_dir)):
            full_path = os.path.join(root_dir, d)
            if os.path.isdir(full_path):
                # Quick check if a melody file likely exists to potentially speed up filtering
                if glob.glob(os.path.join(full_path, '*melody*.mid')):
                     self.song_dirs.append(full_path)
        print(f"Found {len(self.song_dirs)} potential song directories with melody files.")

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, idx):
        song_dir = self.song_dirs[idx]
        # Use the wildcard '*' as you found necessary
        melody_files = glob.glob(os.path.join(song_dir, '*melody.mid'))
        # Use the wildcard '*' as you found necessary
        all_harmony_files = sorted([f for f in glob.glob(os.path.join(song_dir, '*harmony*.mid'))
                                  if f not in melody_files]) # Sort for consistency

        # --- Return dummy if no melody ---
        if not melody_files:
            print(f"Warning: No melody found in {song_dir}. Skipping.")
            dummy_seq = [self.pad_id] * self.max_seq_len
            return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        melody_file = melody_files[0] # Assume only one melody file

        # --- Check for corrupt melody early ---
        try:
            # Try loading melody to catch immediate parsing errors
            _ = pretty_midi.PrettyMIDI(melody_file)
        except Exception as e:
            print(f"Warning: Could not parse melody {melody_file}: {e}. Skipping song.")
            dummy_seq = [self.pad_id] * self.max_seq_len
            return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)


        actual_num_harmony_voices = len(all_harmony_files)

        # --- Handle number of harmony voices ---
        if actual_num_harmony_voices == 0:
            # Still return dummy if no harmony parts are found, as the model needs harmony data
            print(f"Warning: No harmony files found in {song_dir}. Skipping.")
            dummy_seq = [self.pad_id] * self.max_seq_len
            return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        elif actual_num_harmony_voices > self.max_voices:
            # --- MODIFICATION START ---
            # print(f"Info: Found {actual_num_harmony_voices} harmony voices in {song_dir}. Using first {self.max_voices}.")
            harmony_files_to_use = all_harmony_files[:self.max_voices] # Take only the first MAX_VOICES
            num_voices_for_token = self.max_voices # Use MAX_VOICES for the VOICE token
            # --- MODIFICATION END ---
        else:
            harmony_files_to_use = all_harmony_files # Use all found harmony files
            num_voices_for_token = actual_num_harmony_voices # Use the actual count for the VOICE token

        # Construct the voice token string based on the number of voices *used*
        voice_token_str = f"VOICE_{num_voices_for_token}"
        voice_token_id = self.token_to_id.get(voice_token_str, self.token_to_id[UNKNOWN_TOKEN])
        if voice_token_id == self.token_to_id[UNKNOWN_TOKEN]:
             print(f"CRITICAL WARNING: Voice token '{voice_token_str}' not in vocabulary! Check MAX_VOICES.")
             # Return dummy data as this is a critical config error
             dummy_seq = [self.pad_id] * self.max_seq_len
             return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        # --- Tokenize ---
        # Pass only the single melody file
        melody_tokens = midi_to_tokens([melody_file], self.token_to_id, self.time_res, self.max_time, self.vel_bins)
        # Pass the potentially truncated list of harmony files
        harmony_tokens = midi_to_tokens(harmony_files_to_use, self.token_to_id, self.time_res, self.max_time, self.vel_bins)

        # --- Check if tokenization failed (returned empty lists) ---
        if not melody_tokens or not harmony_tokens:
             print(f"Warning: Tokenization failed for melody or harmony in {song_dir}. Skipping.")
             dummy_seq = [self.pad_id] * self.max_seq_len
             return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        # --- Combine and Pad/Truncate ---
        # Format: SOS + VOICE + MELODY + SEP + HARMONY + EOS
        input_sequence = [self.sos_id] + [voice_token_id] + melody_tokens + [self.sep_id] + harmony_tokens + [self.eos_id]

        # Truncate if too long
        if len(input_sequence) > self.max_seq_len:
            input_sequence = input_sequence[:self.max_seq_len -1] + [self.eos_id] # Ensure EOS is last token if truncated

        # --- DEBUG: Check sequence length before padding ---
        # print(f"DEBUG: Sequence length before padding: {len(input_sequence)}")

        # Target sequence is shifted input
        target_sequence = input_sequence[1:]
        input_sequence = input_sequence[:-1] # Input stops before the final target token

        # Pad sequences
        input_padding_len = self.max_seq_len - len(input_sequence)
        target_padding_len = self.max_seq_len - len(target_sequence)

        input_sequence += [self.pad_id] * input_padding_len
        target_sequence += [self.pad_id] * target_padding_len

        # --- DEBUG: Check for invalid indices ---
        # Check if any index is >= VOCAB_SIZE or negative AFTER creating tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)
        try:
            input_tensor = torch.tensor(input_sequence, dtype=torch.long)
            target_tensor = torch.tensor(target_sequence, dtype=torch.long)

            # Check input tensor validity
            if torch.any(input_tensor >= VOCAB_SIZE) or torch.any(input_tensor < 0):
                print(f"\nCRITICAL ERROR in {song_dir}: Invalid token index in input_tensor!")
                print(f"Max index: {input_tensor.max()}, Min index: {input_tensor.min()}, Vocab Size: {VOCAB_SIZE}")
                # Optionally print the problematic sequence part
                invalid_indices = torch.where((input_tensor >= VOCAB_SIZE) | (input_tensor < 0))[0]
                print(f"Problematic input indices positions: {invalid_indices}")
                print(f"Problematic input token values: {input_tensor[invalid_indices]}")
                # Return dummy data or raise error to stop
                dummy_seq = [self.pad_id] * self.max_seq_len
                return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

            # Check target tensor validity (ignoring padding)
            valid_target_indices = target_tensor[target_tensor != self.pad_id]
            if len(valid_target_indices) > 0 and (torch.any(valid_target_indices >= VOCAB_SIZE) or torch.any(valid_target_indices < 0)):
                 print(f"\nCRITICAL ERROR in {song_dir}: Invalid token index in target_tensor!")
                 print(f"Max index: {valid_target_indices.max()}, Min index: {valid_target_indices.min()}, Vocab Size: {VOCAB_SIZE}")
                 # Optionally print the problematic sequence part
                 invalid_indices = torch.where(((target_tensor >= VOCAB_SIZE) | (target_tensor < 0)) & (target_tensor != self.pad_id))[0]
                 print(f"Problematic target indices positions: {invalid_indices}")
                 print(f"Problematic target token values: {target_tensor[invalid_indices]}")
                 # Return dummy data or raise error to stop
                 dummy_seq = [self.pad_id] * self.max_seq_len
                 return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)

        except Exception as e:
             print(f"\nERROR during tensor creation/validation for {song_dir}: {e}")
             dummy_seq = [self.pad_id] * self.max_seq_len
             return torch.tensor(dummy_seq, dtype=torch.long), torch.tensor(dummy_seq, dtype=torch.long)
        # --- END VALIDATION CHECKS ---

        # if torch.any(input_tensor >= VOCAB_SIZE) or torch.any(input_tensor < 0):
        #     print(f"CRITICAL WARNING: Invalid token index in input_tensor for {song_dir}!")
        #     print(f"Max index: {input_tensor.max()}, Min index: {input_tensor.min()}, Vocab Size: {VOCAB_SIZE}")
        #     # Optionally return dummy here or raise error
        # if torch.any(target_tensor >= VOCAB_SIZE) or torch.any(target_tensor < 0):
        #     # Ignore padding index for the check
        #     valid_target_indices = target_tensor[target_tensor != self.pad_id]
        #     if len(valid_target_indices) > 0 and (torch.any(valid_target_indices >= VOCAB_SIZE) or torch.any(valid_target_indices < 0)):
        #          print(f"CRITICAL WARNING: Invalid token index in target_tensor for {song_dir}!")
        #          print(f"Max index: {valid_target_indices.max()}, Min index: {valid_target_indices.min()}, Vocab Size: {VOCAB_SIZE}")
        #          # Optionally return dummy here or raise error

        # Sanity check lengths
        assert len(input_sequence) == self.max_seq_len, f"Input length mismatch: {len(input_sequence)}"
        assert len(target_sequence) == self.max_seq_len, f"Target length mismatch: {len(target_sequence)}"


        return input_tensor, target_tensor # Return tensors directly

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

        # --- MODIFICATION HERE ---
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=ffn_hid_dim, dropout=dropout,
                                                batch_first=True, # Keep batch_first=True
                                                norm_first=True)  # *** ADD THIS ***
        # --- END MODIFICATION ---
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(emb_size, num_tokens)

    def forward(self, src, tgt_mask=None, src_padding_mask=None):
        # src shape: [batch_size, seq_len]
        # tgt_mask: [seq_len, seq_len] (causal mask)
        # src_padding_mask: [batch_size, seq_len] (True where padded)
        try:
            src_emb_tokens = self.token_embedding(src)
            if torch.isnan(src_emb_tokens).any() or torch.isinf(src_emb_tokens).any():
                print("\nCRITICAL WARNING: NaN/Inf detected in TOKEN EMBEDDINGS!")
                # Find which input tokens caused this
                problematic_batch_indices, problematic_seq_indices = torch.where(torch.isnan(src_emb_tokens).any(dim=-1) | torch.isinf(src_emb_tokens).any(dim=-1))
                if len(problematic_batch_indices) > 0:
                        print(f"Problematic input token IDs at first problematic position ({problematic_batch_indices[0]}, {problematic_seq_indices[0]}): {src[problematic_batch_indices[0], problematic_seq_indices[0]]}")
                # Depending on severity, you might want to raise an error here
        except IndexError as e:
            print(f"\nCRITICAL ERROR: IndexError during embedding lookup! Likely invalid token index in src.")
            print(f"Input (src) shape: {src.shape}, Max index in src: {src.max()}, Min index in src: {src.min()}")
            raise e # Reraise the error
        src_emb = self.positional_encoding(self.token_embedding(src))
        # src_emb shape: [batch_size, seq_len, emb_size]
        if torch.isnan(src_emb).any() or torch.isinf(src_emb).any():
            print("\nCRITICAL WARNING: NaN/Inf detected AFTER POSITIONAL ENCODING!")
            # This suggests the positional encoding math might be unstable, or embeddings were already bad

        # --- ADD ACTIVATION STATS CHECK ---
        print(f"\nDEBUG (Forward Pass): Stats for src_emb before TransformerDecoder:")
        print(f"  Shape: {src_emb.shape}")
        print(f"  Min: {src_emb.min():.4f}, Max: {src_emb.max():.4f}, Mean: {src_emb.mean():.4f}, Std: {src_emb.std():.4f}")
        # --- END ACTIVATION STATS CHECK --


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

    for i, (src, tgt) in enumerate(tqdm(dataloader, desc="Training")): # Add index 'i' for batch tracking

         # --- ADDED flush=True to all prints in this block ---
        if i == 0 and epoch == 1: # Combine checks for efficiency
            print(f"\n--- Inspecting First Batch (Epoch {epoch}, Index {i}) ---", flush=True)
            print(f"Batch src shape: {src.shape}", flush=True)
            print(f"Batch tgt shape: {tgt.shape}", flush=True)
            print(f"First sequence (src) start: {src[0, :50]}...", flush=True)
            print(f"First sequence (tgt) start: {tgt[0, :50]}...", flush=True)
            try:
                # Ensure id_to_token is accessible if needed here
                global id_to_token # Or pass it into train_epoch
                print(f"First seq (src) tokens: {[id_to_token.get(t.item(), '?') for t in src[0, :50]]}...", flush=True)
            except NameError:
                pass # Silently ignore if id_to_token isn't available
            print(f"Min/Max values in src: {src.min()}, {src.max()}", flush=True)
            # Ensure pad_idx is accessible
            print(f"Min/Max values in tgt (excluding PAD={pad_idx}): {tgt[tgt != pad_idx].min() if torch.any(tgt != pad_idx) else 'N/A'}, {tgt[tgt != pad_idx].max() if torch.any(tgt != pad_idx) else 'N/A'}", flush=True)
            # print(f"Device: {src.device}", flush=True) # Device before moving

        src, tgt = src.to(DEVICE), tgt.to(DEVICE) # Move data AFTER inspection if needed
        seq_len = src.size(1)
        tgt_mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
        src_padding_mask = (src == pad_idx)

        # --- ADDED flush=True to all prints in this block ---
        if i == 0 and epoch == 1:
            print("\n--- Checking Masks (Batch 0) ---", flush=True)
            print(f"tgt_mask shape: {tgt_mask.shape}, dtype: {tgt_mask.dtype}", flush=True)
            print(f"src_padding_mask shape: {src_padding_mask.shape}, dtype: {src_padding_mask.dtype}", flush=True)
            print(f"--- End Mask Checks ---", flush=True)

        optimizer.zero_grad()

        logits = model(src, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)

        # --- ADD THIS CHECK ---
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"\nCRITICAL WARNING: NaN or Inf detected in model logits at batch index {i}!")
            # Optionally print parts of the input that caused it
            # print("Sample problematic src:", src[0, :30])
            # print("Sample problematic tgt:", tgt[0, :30])
            # Decide how to handle: skip batch? stop training?
            # For now, let's just continue and see if loss becomes NaN
            # If loss becomes NaN consistently after this, the logits are the source.

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if torch.isnan(loss):
             print(f"\nCRITICAL WARNING: NaN detected in loss calculation at batch index {i}!")
             # If the logit check above didn't trigger, the issue might be specific
             # interaction between logits and targets in the loss function.
             raise RuntimeError(f"NaN loss detected at batch {i}. Stopping training.") # Stop if loss is NaN


        loss.backward() # The error you saw happens here or within this call


        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"\nDEBUG: Grad Norm BEFORE clipping (Batch {i}): {total_norm:.4f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()

    # Avoid division by zero if num_batches is 0 (though unlikely)
    if num_batches == 0:
         return 0.0
    return total_loss / num_batches

# --- Main Execution ---

if __name__ == "__main__":
    # --- 1. Build or Load Vocabulary ---
    if not os.path.exists(TOKENIZER_PATH):
        print("Building vocabulary...")
        token_to_id, id_to_token = build_vocabulary(
            DATASET_ROOT, TIME_RESOLUTION, MAX_TIME_SHIFT, VELOCITY_BINS, MAX_VOICES
        )
        save_tokenizer(token_to_id, id_to_token, TOKENIZER_PATH)
        print("Vocabulary built and saved.")
    else:
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        token_to_id, id_to_token = load_tokenizer(TOKENIZER_PATH)

    VOCAB_SIZE = len(token_to_id)
    print(f"Determined Vocabulary Size (VOCAB_SIZE): {VOCAB_SIZE}")
    # --- ASSERTION: Ensure VOCAB_SIZE is sufficient ---
    # Check if the vocabulary seems reasonably sized based on components
    # This is a basic heuristic check
    min_expected_size = 128 + VELOCITY_BINS + int(MAX_TIME_SHIFT / TIME_RESOLUTION) + MAX_VOICES + 5 # Notes + Vel + Time + Voices + Special
    assert VOCAB_SIZE >= min_expected_size, f"FATAL: VOCAB_SIZE ({VOCAB_SIZE}) seems too small. Expected at least {min_expected_size} based on config. Check tokenization/vocab building."
    # You could also add the check based on max observed index if you run into issues again:
    # max_observed_index = 246 # Or derive this programmatically if needed
    # assert VOCAB_SIZE > max_observed_index, f"FATAL: VOCAB_SIZE ({VOCAB_SIZE}) is not greater than max observed index ({max_observed_index})!"

    PAD_IDX = token_to_id[PAD_TOKEN] # Set PAD_IDX *after* loading vocabulary

    # --- 2. Create Datasets and Dataloaders ---
    # Consider splitting your data into train/validation sets later for better evaluation
    print("Creating dataset...")
    train_dataset = MIDIDataset(
        root_dir=DATASET_ROOT,
        token_to_id=token_to_id,
        max_seq_len=MAX_SEQ_LEN,
        max_voices=MAX_VOICES,
        time_res=TIME_RESOLUTION,
        max_time=MAX_TIME_SHIFT,
        vel_bins=VELOCITY_BINS
    )
    print(f"Dataset size: {len(train_dataset)} items.")

    print("Creating dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True) # Adjust num_workers based on your OS/setup

    # --- 3. Initialize Model, Loss, Optimizer, Scheduler ---
    print("Initializing model...")
    model = MusicTransformerDecoder(
        num_tokens=VOCAB_SIZE,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        ffn_hid_dim=FFN_HID_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT, # Dropout is active during training
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore padding in loss calculation
    optimizer = optim.AdamW(model.parameters(), lr=LR) # Use the initial LR (e.g., 1e-5 if you lowered it)

    # --- Learning Rate Warm-up Scheduler ---
    # Calculate total expected steps if needed for more complex schedulers,
    # but for linear warmup, just need warmup_steps.
    try:
        num_training_steps_per_epoch = len(train_dataloader)
        if num_training_steps_per_epoch == 0:
             raise ValueError("DataLoader is empty! Check dataset/filtering.")
    except Exception as e:
         print(f"ERROR getting DataLoader length: {e}")
         # Set a default or raise error, depending on how you want to handle empty datasets
         num_training_steps_per_epoch = 1 # Avoid division by zero, but training will likely fail

    warmup_epochs = 1 # How many epochs to warm up over (adjust as needed)
    warmup_steps = warmup_epochs * num_training_steps_per_epoch
    print(f"Scheduler: Warming up learning rate linearly for {warmup_steps} steps...")

    def lr_lambda(current_step):
        # current_step is 0-indexed
        if current_step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(current_step + 1) / float(max(1.0, warmup_steps)) # Add 1 to current_step for 1-based counting in warmup fraction
        # After warmup, maintain the base LR (or add decay here later)
        return 1.0 # Maintain base LR after warmup

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # --- End Scheduler ---

    print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using Device: {DEVICE}")
    print(f"Initial Learning Rate (set in optimizer): {LR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Sequence Length: {MAX_SEQ_LEN}")

    # --- Optional: Enable anomaly detection for debugging NaNs ---
    # torch.autograd.set_detect_anomaly(True) # Uncomment if still debugging NaNs

    # --- 4. Training Loop ---
    print("Starting Training...")
    global_step = 0 # Keep track of total optimizer steps
    for epoch in range(1, NUM_EPOCHS + 1):

        model.train() # Set model to training mode for each epoch
        total_loss = 0
        num_batches = len(train_dataloader) # Use train_dataloader here

        # Use tqdm for progress bar
        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for i, batch_data in enumerate(batch_iterator):

            # --- Handle potential dummy data if __getitem__ returns it ---
            # This depends on how you implemented error handling in __getitem__
            # Assuming it returns tensors of PAD_IDX on error
            if batch_data is None: # If using a collate_fn that returns None
                print(f"Warning: Skipping None batch at epoch {epoch}, index {i}")
                continue

            src, tgt = batch_data
            # Check if the batch consists only of padding (might happen if many errors occurred)
            if torch.all(src == PAD_IDX):
                 print(f"Warning: Skipping batch of only PAD tokens at epoch {epoch}, index {i}")
                 continue

            # --- Inspect First Batch (keep if debugging needed) ---
            # if i == 0 and epoch == 1:
            #     print(f"\n--- Inspecting First Batch (Epoch {epoch}, Index {i}) ---")
            #     print(f"Batch src shape: {src.shape}")
            #     print(f"Batch tgt shape: {tgt.shape}")
            #     print(f"First sequence (src) start: {src[0, :50]}...")
            #     print(f"Min/Max values in src: {src.min()}, {src.max()}")
            #     print(f"Min/Max values in tgt (excluding PAD={PAD_IDX}): {tgt[tgt != PAD_IDX].min() if torch.any(tgt != PAD_IDX) else 'N/A'}, {tgt[tgt != PAD_IDX].max() if torch.any(tgt != PAD_IDX) else 'N/A'}")
            #     print(f"Device: {src.device}") # Should be CPU initially from dataloader
            #     print(f"--- End First Batch Inspection ---")

            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            seq_len = src.size(1)
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(DEVICE)
            src_padding_mask = (src == PAD_IDX)

            # Zero gradients BEFORE the forward pass and loss calculation
            optimizer.zero_grad()

            # --- Forward Pass ---
            logits = model(src, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)

            # --- Logit Check (keep if debugging) ---
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\nCRITICAL WARNING: NaN or Inf detected in model logits at epoch {epoch}, batch index {i}!")
                # Consider raising error or skipping batch if this happens after initial warmup
                # For now, just print the warning
                # raise RuntimeError(f"NaN/Inf in logits detected at E{epoch} B{i}")

            # --- Loss Calculation ---
            # Reshape for CrossEntropyLoss: needs [N, C] and [N]
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))

            # --- Loss Check ---
            if torch.isnan(loss):
                 print(f"\nCRITICAL WARNING: NaN detected in loss calculation at epoch {epoch}, batch index {i}!")
                 # Print details that might help diagnose
                 # print("Logits sample (first token):", logits[0, 0, :10])
                 # print("Target sample (first token):", tgt[0,0])
                 raise RuntimeError(f"NaN loss detected at E{epoch} B{i}. Stopping training.")

            # --- Backward Pass ---
            loss.backward() # Compute gradients

            # --- Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD) # Use the global CLIP_GRAD value

            # --- Optimizer Step ---
            optimizer.step() # Update weights based on computed gradients and current learning rate

            # --- Scheduler Step ---
            # Crucially, step the scheduler AFTER the optimizer step
            scheduler.step()
            global_step += 1 # Increment global step counter

            # --- Logging ---
            total_loss += loss.item()
            # Update tqdm description with current loss and LR
            current_lr = scheduler.get_last_lr()[0] # Get current LR from scheduler
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})
            # --- End Batch Loop ---

        # --- End of Epoch ---
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            final_lr_epoch = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{NUM_EPOCHS} Summary: Average Training Loss: {avg_loss:.4f}, Final LR for Epoch: {final_lr_epoch:.8f}")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS} Summary: No batches processed.")
            avg_loss = float('nan') # Assign NaN if no batches were processed

        # --- 5. Save Model Checkpoint ---
        if epoch % 5 == 0 or epoch == NUM_EPOCHS: # Save every 5 epochs and at the end
            checkpoint_path = f"model_epoch_{epoch}.pth"
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
                    'loss': avg_loss,
                    # Include tokenizer info for reproducibility
                    'token_to_id': token_to_id,
                    'id_to_token': id_to_token,
                    'config': { # Save key hyperparameters used for this training run
                        'vocab_size': VOCAB_SIZE,
                        'emb_size': EMB_SIZE,
                        'nhead': NHEAD,
                        'num_layers': NUM_LAYERS,
                        'ffn_hid_dim': FFN_HID_DIM,
                        'max_seq_len': MAX_SEQ_LEN,
                        'dropout': DROPOUT, # The dropout used during training
                        'time_res': TIME_RESOLUTION,
                        'max_time': MAX_TIME_SHIFT,
                        'vel_bins': VELOCITY_BINS,
                        'max_voices': MAX_VOICES,
                        'pad_idx': PAD_IDX,
                        'lr_initial': LR, # Initial LR before scheduling
                        'batch_size': BATCH_SIZE,
                        'clip_grad': CLIP_GRAD,
                        'warmup_epochs': warmup_epochs
                    }
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint at epoch {epoch}: {e}")
        # --- End Epoch Loop ---

    print("Training Finished.")
    # --- 6. Save Final Model State Dict (Optional) ---
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Final model state_dict saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving final model state_dict: {e}")