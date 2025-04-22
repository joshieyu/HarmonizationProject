#!/usr/bin/env python3
"""
MIDI Strict Monophonic Voice Extraction Script (using pretty_midi)

Extracts:
1. A strictly monophonic melody track (*_melody.mid).
2. ONLY strictly monophonic harmony tracks (*_harmony1.mid, *_harmony2.mid, ...).

Skips files that do not contain at least one monophonic melody AND
at least one *other* monophonic track to serve as harmony.
"""

import os
import glob
import pretty_midi
import numpy as np
from tqdm import tqdm
import argparse

# --- Configuration ---
MONOPHONIC_TOLERANCE = 1e-4

# --- Helper Functions (keep is_monophonic, get_average_pitch) ---

def is_monophonic(notes, tolerance=MONOPHONIC_TOLERANCE):
    """ Checks if a list of pretty_midi notes represents a monophonic track. """
    if not notes: return True
    sorted_notes = sorted(notes, key=lambda n: (n.start, n.end))
    last_note_end = -1.0
    for note in sorted_notes:
        if note.start < last_note_end - tolerance: return False
        last_note_end = max(last_note_end, note.end)
    return True

def get_average_pitch(notes):
    """ Calculates the average pitch of notes in a list. """
    if not notes: return 0
    pitches = [note.pitch for note in notes]
    if not pitches: return 0
    return sum(pitches) / len(pitches)

def extract_strict_monophonic_voices(midi_file, output_dir):
    """
    Extracts monophonic melody and only monophonic harmony tracks.

    Args:
        midi_file (str): Path to the input MIDI file.
        output_dir (str): Base directory to save the extracted files.

    Returns:
        list: Paths to the created melody and harmony files, or empty list on failure/skip.
    """
    midi_name = os.path.splitext(os.path.basename(midi_file))[0]
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        instruments_with_notes = [(i, inst) for i, inst in enumerate(midi_data.instruments) if inst.notes]

        if not instruments_with_notes:
            # print(f"  Skipping {midi_name}: No instruments with notes found.")
            return []

        # 1. Identify all monophonic tracks and calculate their avg pitch
        monophonic_tracks = [] # Store tuples: (avg_pitch, original_idx, instrument)
        for original_idx, instrument in instruments_with_notes:
            if is_monophonic(instrument.notes, tolerance=MONOPHONIC_TOLERANCE):
                avg_pitch = get_average_pitch(instrument.notes)
                monophonic_tracks.append((avg_pitch, original_idx, instrument))

        # 2. Check if *any* monophonic track was found
        if not monophonic_tracks:
            print(f"  Skipping {midi_name}: No monophonic tracks found at all.")
            return []

        # 3. Select the best melody candidate (highest average pitch among monophonic)
        monophonic_tracks.sort(key=lambda x: x[0], reverse=True) # Sort descending by pitch
        _, best_melody_original_idx, best_melody_instrument = monophonic_tracks[0]

        # 4. Identify *other* tracks that are *also* monophonic (for harmony)
        monophonic_harmony_tracks = [] # Store tuples: (original_idx, instrument)
        for avg_pitch, original_idx, instrument in monophonic_tracks:
            if original_idx != best_melody_original_idx:
                 monophonic_harmony_tracks.append((original_idx, instrument))

        # 5. Check if at least one suitable *monophonic* harmony track exists
        if not monophonic_harmony_tracks:
            print(f"  Skipping {midi_name}: Monophonic melody found, but no other suitable *monophonic* harmony tracks.")
            return []

        # --- Proceed with saving ---
        midi_output_dir = os.path.join(output_dir, midi_name)
        os.makedirs(midi_output_dir, exist_ok=True)
        output_files = []

        # 6. Save the Monophonic Melody File
        melody_midi = pretty_midi.PrettyMIDI()
        melody_instrument_out = pretty_midi.Instrument(program=0, is_drum=False, name="Melody")
        melody_instrument_out.notes.extend(best_melody_instrument.notes)
        melody_midi.instruments.append(melody_instrument_out)
        melody_output_path = os.path.join(midi_output_dir, f"{midi_name}_melody.mid")
        try:
            melody_midi.write(melody_output_path)
            output_files.append(melody_output_path)
        except Exception as write_e:
            print(f"  ERROR writing melody file {melody_output_path}: {write_e}")
            return [] # Fail if melody can't be written

        # 7. Save Separate *Monophonic* Harmony Files
        # Sort harmony tracks (e.g., by original index) for consistent naming, though not strictly necessary
        monophonic_harmony_tracks.sort(key=lambda x: x[0])
        harmony_file_counter = 1
        for original_idx, harmony_instrument in monophonic_harmony_tracks:
            harmony_midi = pretty_midi.PrettyMIDI()
            harmony_instrument_out = pretty_midi.Instrument(program=0, is_drum=False, name=f"Harmony {harmony_file_counter}")
            harmony_instrument_out.notes.extend(harmony_instrument.notes)
            harmony_midi.instruments.append(harmony_instrument_out)
            harmony_output_path = os.path.join(midi_output_dir, f"{midi_name}_harmony{harmony_file_counter}.mid")
            try:
                harmony_midi.write(harmony_output_path)
                output_files.append(harmony_output_path)
                harmony_file_counter += 1
            except Exception as write_e:
                 print(f"  ERROR writing harmony file {harmony_output_path}: {write_e}")
                 # Continue saving others, but maybe log the failure more prominently

        # Return list of successfully created files
        return output_files

    except ValueError as ve:
         print(f"  ERROR processing {midi_name} (likely timing issue): {ve}")
         return []
    except Exception as e:
        print(f"  ERROR processing {midi_name}: {type(e).__name__} - {e}")
        return []


def process_midi_folder(input_folder, output_folder):
    """ Processes all MIDI files in a folder. """
    os.makedirs(output_folder, exist_ok=True)
    midi_files = glob.glob(os.path.join(input_folder, "*.mid")) + glob.glob(os.path.join(input_folder, "*.midi"))
    if not midi_files: print(f"No MIDI files found in {input_folder}"); return

    print(f"Found {len(midi_files)} MIDI files. Processing (strict monophonic voice extraction)...")
    processed_count = 0
    skipped_count = 0

    for midi_file in tqdm(midi_files, desc="Processing MIDI"):
        # Use the updated function name
        output_files = extract_strict_monophonic_voices(midi_file, output_folder)
        # Check if at least melody and one harmony were produced (meaning suitable file found)
        if len(output_files) >= 2:
            processed_count += 1
        else:
            skipped_count += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed (found mono melody + >=1 mono harmony): {processed_count}")
    print(f"Skipped/Errored (unsuitable structure or error): {skipped_count}")
    print(f"Output files generated in subdirectories within: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract STRICTLY monophonic melody and harmony tracks from MIDI files.")
    parser.add_argument("--input", "-i", default="HarmDataset", help="Input folder")
    parser.add_argument("--output", "-o", default="ProcessedDatasetStrict", help="Output folder") # Changed default output folder name
    parser.add_argument("--tolerance", "-t", type=float, default=MONOPHONIC_TOLERANCE, help=f"Max overlap for monophonic check (secs, default: {MONOPHONIC_TOLERANCE})")
    args = parser.parse_args()

    MONOPHONIC_TOLERANCE = args.tolerance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, args.input)
    output_folder = os.path.join(script_dir, args.output) # Use the potentially new output name

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input folder not found: {input_folder}")
    else:
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Monophonic Tolerance: {MONOPHONIC_TOLERANCE} seconds")
        process_midi_folder(input_folder, output_folder)