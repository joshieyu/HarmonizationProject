#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.optim as optim # Not strictly needed for inference, but maybe for loading optimizer state if desired

import pretty_midi
import numpy as np
import os
import glob
import math
import pickle
from collections import Counter
from tqdm import tqdm # For progress bars
import argparse
import sys

# --- Global Constants (Defaults/Fallbacks) ---
# These are used if specific config values are missing from the checkpoint
DEFAULT_TIME_RESOLUTION = 0.02
DEFAULT_MAX_TIME_SHIFT = 2.0
DEFAULT_VELOCITY_BINS = 4
DEFAULT_MAX_VOICES = 10 # Default max voices expected if not in config

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"
UNKNOWN_TOKEN = "<unk>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---

def velocity_to_bin(velocity, bins=DEFAULT_VELOCITY_BINS):
    """Maps MIDI velocity (0-127) to a discrete bin."""
    if velocity == 0: return 0
    # Ensure bins is positive to avoid division by zero
    if bins <= 0: bins = 1
    return min(bins - 1, int(velocity / (128.0 / bins)))

def time_to_bin(time_delta, resolution=DEFAULT_TIME_RESOLUTION, max_time=DEFAULT_MAX_TIME_SHIFT):
    """Maps a time delta (seconds) to a discrete time shift token bin."""
    # Ensure resolution is positive
    if resolution <= 0: resolution = 0.01 # Small default if invalid
    num_time_bins = int(max_time / resolution)
    if num_time_bins <= 0: num_time_bins = 1 # Ensure at least one bin

    if time_delta >= max_time:
        return num_time_bins - 1
    elif time_delta < 0:
        return 0
    else:
        return int(time_delta / resolution)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        seq_len = token_embedding.size(1)
        # Ensure maxlen is large enough, otherwise narrow might fail
        maxlen_buffer = self.pos_embedding.size(1)
        if seq_len > maxlen_buffer:
             # This shouldn't happen if max_seq_len is set correctly during training/init
             # but handle defensively. You might want to regenerate PE if needed.
             print(f"Warning: Sequence length {seq_len} exceeds PositionalEncoding maxlen {maxlen_buffer}. Truncating.")
             seq_len = maxlen_buffer
             token_embedding = token_embedding[:, :seq_len, :] # Truncate input embedding

        pos_emb = self.pos_embedding.narrow(1, 0, seq_len)
        return self.dropout(token_embedding + pos_emb)

# --- Transformer Model Definition (Must match the one used for training) ---
class MusicTransformerDecoder(nn.Module):
    def __init__(self, num_tokens, emb_size, nhead, ffn_hid_dim, num_layers, dropout, max_seq_len):
        super(MusicTransformerDecoder, self).__init__()
        # Add padding_idx to embedding layer if needed/used during training
        # padding_idx = token_to_id[PAD_TOKEN] # Requires token_to_id lookup here
        self.token_embedding = nn.Embedding(num_tokens, emb_size) # Add padding_idx=pad_idx if used
        self.positional_encoding = PositionalEncoding(emb_size, dropout, max_seq_len) # Use max_seq_len here

        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead,
            dim_feedforward=ffn_hid_dim, dropout=dropout,
            batch_first=True,
            norm_first=True # Assumes you trained with norm_first=True for stability
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(emb_size, num_tokens)

    def forward(self, src, tgt_mask=None, src_padding_mask=None):
        if torch.any(src >= self.token_embedding.num_embeddings) or torch.any(src < 0):
            raise IndexError(f"Input indices out of bounds! Max: {src.max()}, Min: {src.min()}, Vocab: {self.token_embedding.num_embeddings}")

        src_emb = self.positional_encoding(self.token_embedding(src))
        if torch.isnan(src_emb).any(): raise RuntimeError("NaN detected after Embedding/PE")

        output = self.transformer_decoder(
            tgt=src_emb, memory=src_emb,
            tgt_mask=tgt_mask, memory_mask=None,
            tgt_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask
        )
        if torch.isnan(output).any(): raise RuntimeError("NaN detected after Transformer layers")

        logits = self.output_linear(output)
        if torch.isnan(logits).any(): raise RuntimeError("NaN detected in final logits")

        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- Weight Initialization Function (Optional but good practice) ---
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight) # Or normal_
        # If using padding_idx, ensure that row is zero:
        # if m.padding_idx is not None:
        #     with torch.no_grad(): m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, nn.LayerNorm):
         nn.init.ones_(m.weight)
         nn.init.zeros_(m.bias)

# --- MIDI Processing Functions ---

def midi_melody_to_tokens(melody_midi_path, token_to_id, config):
    """Converts a single melody MIDI file into a sequence of token IDs for the prompt."""
    time_res = config['time_res']
    max_time = config['max_time']
    vel_bins = config['vel_bins']
    unk_id = token_to_id.get(UNKNOWN_TOKEN, 0) # Use PAD if UNK not present

    tokens = []
    try:
        midi_data = pretty_midi.PrettyMIDI(melody_midi_path)
        if not midi_data.instruments:
            print(f"Warning: No instruments found in {melody_midi_path}")
            return None
        instrument = midi_data.instruments[0] # Assume melody is first instrument
        instrument.notes.sort(key=lambda x: x.start)

        last_event_time = 0.0
        for note in instrument.notes:
            delta_t = note.start - last_event_time
            if delta_t < 0: delta_t = 0

            time_token_val = time_to_bin(delta_t, time_res, max_time)
            note_token_val = note.pitch
            vel_token_val = velocity_to_bin(note.velocity, vel_bins)

            time_token_str = f"TIME_{time_token_val}"
            note_token_str = f"NOTEON_{note_token_val}"
            vel_token_str = f"VEL_{vel_token_val}"

            tokens.append(token_to_id.get(time_token_str, unk_id))
            tokens.append(token_to_id.get(note_token_str, unk_id))
            tokens.append(token_to_id.get(vel_token_str, unk_id))

            last_event_time = note.start

    except Exception as e:
        print(f"Error processing melody MIDI {melody_midi_path}: {e}")
        return None
    return tokens

def tokens_to_separate_midis(token_ids, id_to_token, config, num_voices, output_prefix):
    """Converts token sequence to separate MIDI files for each harmony voice."""
    print(f"Converting tokens to {num_voices} separate MIDI files...")
    if num_voices <= 0: print("Error: Num voices must be > 0."); return

    # Extract necessary info from config
    pad_id = config['pad_idx']
    eos_id = config['token_to_id'].get(EOS_TOKEN) # Get EOS ID
    sep_id = config['token_to_id'].get(SEP_TOKEN) # Get SEP ID
    time_res = config['time_res']
    vel_bins = config['vel_bins']
    unk_token_str = UNKNOWN_TOKEN # Use constant

    if eos_id is None: print("Warning: EOS_TOKEN not found in tokenizer."); eos_id = -1 # Assign invalid ID
    if sep_id is None: print("Warning: SEP_TOKEN not found in tokenizer."); sep_id = -1 # Assign invalid ID

    harmony_notes_by_voice = [[] for _ in range(num_voices)]
    current_time = 0.0
    current_velocity = 64
    harmony_note_counter = 0

    try:
        generation_start_index = token_ids.index(sep_id) + 1
    except ValueError:
        print("CRITICAL WARNING: Separator token not found in generated sequence!")
        print("  Cannot reliably determine start of harmony. Aborting MIDI conversion.")
        return # Stop conversion if SEP is missing

    print(f"Starting harmony note extraction from token index: {generation_start_index}")

    for i in range(generation_start_index, len(token_ids)):
        token_id = token_ids[i]

        if token_id == eos_id: print("Reached EOS token."); break
        if token_id == pad_id: print("Reached PAD token."); break

        token_str = id_to_token.get(token_id, unk_token_str)

        if token_str.startswith("TIME_"):
            try: time_bin = int(token_str.split("_")[1]); current_time += time_bin * time_res
            except: pass # Ignore parsing errors for time
        elif token_str.startswith("VEL_"):
            try:
                vel_bin = int(token_str.split("_")[1])
                current_velocity = max(1, min(127, int((vel_bin + 0.5) * (128.0 / vel_bins))))
            except: pass # Ignore parsing errors for velocity
        elif token_str.startswith("NOTEON_"):
            try:
                pitch = int(token_str.split("_")[1])
                duration = 0.4 # FIXME: Placeholder duration - needs proper handling!
                note_obj = pretty_midi.Note(velocity=current_velocity, pitch=pitch, start=current_time, end=current_time + duration)
                target_voice_index = harmony_note_counter % num_voices
                harmony_notes_by_voice[target_voice_index].append(note_obj)
                harmony_note_counter += 1
            except Exception as e: print(f"Warning: Could not process NOTEON token {token_str}: {e}")

    # --- Save Separate MIDI Files ---
    output_filenames = []
    for voice_idx in range(num_voices):
        if not harmony_notes_by_voice[voice_idx]:
            print(f"Info: No notes generated for harmony voice {voice_idx + 1}. Skipping file.")
            continue

        harmony_midi = pretty_midi.PrettyMIDI()
        harmony_instrument = pretty_midi.Instrument(program=0, is_drum=False, name=f"Generated Harmony {voice_idx + 1}")
        harmony_instrument.notes.extend(harmony_notes_by_voice[voice_idx])
        harmony_midi.instruments.append(harmony_instrument)
        output_filename = f"{output_prefix}_harmony{voice_idx + 1}.mid"
        try:
            harmony_midi.write(output_filename)
            output_filenames.append(output_filename)
            print(f"  Saved: {output_filename}")
        except Exception as e: print(f"  ERROR writing harmony file {output_filename}: {e}")

    print(f"Finished saving {len(output_filenames)} harmony MIDI files.")

# --- Generation Function ---
def generate_sequence(model, tokenizer, prompt_tokens, max_gen_len, temperature, top_p, device):
    """Generates token sequence autoregressively using nucleus sampling."""
    model.eval()
    token_to_id = tokenizer['token_to_id']
    id_to_token = tokenizer['id_to_token']
    pad_id = token_to_id.get(PAD_TOKEN, 0) # Default to 0 if PAD not found
    eos_id = token_to_id.get(EOS_TOKEN, -1) # Use -1 if EOS not found

    generated_tokens = prompt_tokens[:]
    model_max_ctx = model.positional_encoding.pos_embedding.size(1)

    with torch.no_grad():
        for _ in tqdm(range(max_gen_len), desc="Generating Tokens"):
            input_seq = generated_tokens[-model_max_ctx:]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            seq_len = input_tensor.size(1)
            tgt_mask = model.generate_square_subsequent_mask(seq_len).to(device)
            src_padding_mask = (input_tensor == pad_id)

            try:
                 logits = model(input_tensor, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)
            except Exception as e:
                 print(f"\nError during model forward pass in generation: {e}")
                 print(f"Input shape: {input_tensor.shape}")
                 break # Stop generation if model fails

            last_logits = logits[0, -1, :]
            if torch.isnan(last_logits).any():
                 print("\nWarning: NaN detected in logits during generation. Stopping.")
                 break

            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
            else: # Greedy
                probs = torch.softmax(last_logits, dim=-1)

            if top_p > 0.0 and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                if torch.sum(probs) > 0:
                     probs.div_(torch.sum(probs))
                else:
                     # If all probs are zero (e.g., numerical issue or extreme top_p)
                     # fall back to selecting the original most likely token
                     print("\nWarning: All probabilities zeroed out in top-p. Using top-1.")
                     next_token_id = torch.argmax(logits[0, -1, :]).item()
                     generated_tokens.append(next_token_id)
                     continue # Skip multinomial sampling

            next_token_id = torch.multinomial(probs, num_samples=1).squeeze().item()
            generated_tokens.append(next_token_id)

            if next_token_id == eos_id: print("\nEOS token generated."); break
            if next_token_id == pad_id: print("\nWarning: PAD token generated."); break # Should be rare

    return generated_tokens

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separate harmony MIDI files using a trained Transformer model.")
    parser.add_argument("melody_midi", help="Path to the input melody MIDI file.")
    parser.add_argument("num_voices", type=int, help="Number of separate harmony voices to generate.")
    parser.add_argument("--output_prefix", "-o", default="generated_harmony", help="Prefix for the output harmony MIDI files (default: generated_harmony).")
    parser.add_argument("--checkpoint", "-c", default="model_epoch_50.pth", help="Path to the trained model checkpoint (.pth file) (default: model_epoch_50.pth).")
    parser.add_argument("--max_gen_len", type=int, default=512, help="Maximum number of tokens to GENERATE (default: 512).")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature (default: 0.75).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold (default: 0.9).")
    args = parser.parse_args()

    if args.num_voices <= 0: print("ERROR: Number of voices must be > 0."); sys.exit(1)
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}"); os.makedirs(output_dir)

    # --- Load Model and Tokenizer ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint): print(f"ERROR: Checkpoint file not found: {args.checkpoint}"); sys.exit(1)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        config = checkpoint.get('config', {}) # Use empty dict if config missing
        token_to_id = checkpoint['token_to_id']
        id_to_token = checkpoint['id_to_token']
        tokenizer_data = {'token_to_id': token_to_id, 'id_to_token': id_to_token}

        # --- Populate config with defaults if keys are missing ---
        config['time_res'] = config.get('time_res', DEFAULT_TIME_RESOLUTION)
        config['max_time'] = config.get('max_time', DEFAULT_MAX_TIME_SHIFT)
        config['vel_bins'] = config.get('vel_bins', DEFAULT_VELOCITY_BINS)
        config['max_voices'] = config.get('max_voices', DEFAULT_MAX_VOICES)
        config['pad_idx'] = config.get('pad_idx', token_to_id.get(PAD_TOKEN, 0))
        config['vocab_size'] = config.get('vocab_size', len(token_to_id))
        # Add token IDs to config for easy access in helpers
        config['token_to_id'] = token_to_id
        config['id_to_token'] = id_to_token

        # Check if requested voices exceed model's likely capability
        if args.num_voices > config['max_voices']:
             print(f"Warning: Requested {args.num_voices} voices vs model trained with MAX_VOICES={config['max_voices']}.")
        voice_token_str_check = f"VOICE_{args.num_voices}"
        if voice_token_str_check not in token_to_id:
             print(f"ERROR: Voice token '{voice_token_str_check}' not in vocabulary. Cannot generate {args.num_voices} voices.")
             sys.exit(1)

        # --- Initialize model ---
        model = MusicTransformerDecoder(
            num_tokens=config['vocab_size'],
            emb_size=config.get('emb_size', 512), # Provide defaults if missing
            nhead=config.get('nhead', 8),
            ffn_hid_dim=config.get('ffn_hid_dim', 2048),
            num_layers=config.get('num_layers', 6),
            dropout=0.0, # No dropout for inference
            max_seq_len=config.get('max_seq_len', 1024)
        ).to(DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")

    except KeyError as ke: print(f"ERROR: Missing key in checkpoint: {ke}"); sys.exit(1)
    except Exception as e: print(f"ERROR loading checkpoint/model: {type(e).__name__} - {e}"); sys.exit(1)

    # --- Process Melody ---
    print(f"Processing melody MIDI: {args.melody_midi}")
    melody_tokens = midi_melody_to_tokens(args.melody_midi, token_to_id, config)
    if melody_tokens is None: print("Could not process melody. Exiting."); sys.exit(1)

    # --- Construct Prompt ---
    sos_id = token_to_id[SOS_TOKEN]
    sep_id = token_to_id[SEP_TOKEN]
    voice_token_id = token_to_id[f"VOICE_{args.num_voices}"]
    prompt_tokens = [sos_id] + [voice_token_id] + melody_tokens + [sep_id]
    prompt_len = len(prompt_tokens)
    print(f"Prompt constructed ({prompt_len} tokens).")

    model_max_len = config.get('max_seq_len', 1024) # Use loaded or default
    if prompt_len >= model_max_len:
        print(f"ERROR: Prompt length ({prompt_len}) >= model max length ({model_max_len})."); sys.exit(1)

    # --- Generate Sequence ---
    print(f"Generating sequence (max_gen_len={args.max_gen_len}, temp={args.temperature}, top_p={args.top_p})...")
    generated_token_ids = generate_sequence(
        model=model, tokenizer=tokenizer_data, prompt_tokens=prompt_tokens,
        max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p, device=DEVICE
    )
    print(f"Generation finished ({len(generated_token_ids)} total tokens).")

    # --- Convert Tokens to MIDI ---
    tokens_to_separate_midis(
        token_ids=generated_token_ids, id_to_token=id_to_token, config=config,
        num_voices=args.num_voices, output_prefix=args.output_prefix
    )

    print("\nInference complete.")