#!/usr/bin/env python3
"""
MIDI Voice Extraction Preprocessing Script

This script processes polyphonic MIDI files by splitting them into separate voice files.
The top-most voice (highest pitch) is labeled as the melody, and lower voices are
labeled as harmony1, harmony2, etc.
"""

import os
import glob
from collections import defaultdict
from music21 import converter, stream, note, chord, instrument

def extract_voices(midi_file, output_dir):
    """
    Extract individual voices from a polyphonic MIDI file.
    
    Args:
        midi_file (str): Path to the MIDI file
        output_dir (str): Directory to save the extracted voices
        
    Returns:
        list: Paths to the created voice files
    """
    try:
        print(f"Processing {os.path.basename(midi_file)}...")
        
        # Parse the MIDI file
        midi_data = converter.parse(midi_file)
        
        # Create a directory for this specific MIDI file's voices
        midi_name = os.path.splitext(os.path.basename(midi_file))[0]
        midi_output_dir = os.path.join(output_dir, midi_name)
        os.makedirs(midi_output_dir, exist_ok=True)
        
        # Get all parts/voices from the MIDI file
        parts = []
        
        # Check if the score has parts
        if midi_data.hasPartLikeStreams():
            parts = midi_data.parts
        # If not, try to identify voices from a flat structure
        else:
            # Extract all notes and create voice streams based on pitch
            all_notes = midi_data.flat.notes
            
            # Group notes by their start time
            note_dict = defaultdict(list)
            for note_obj in all_notes:
                if note_obj.offset not in note_dict:
                    note_dict[note_obj.offset] = []
                
                # Handle both individual notes and chords
                if isinstance(note_obj, chord.Chord):
                    for pitch in note_obj.pitches:
                        n = note.Note(pitch)
                        n.duration = note_obj.duration
                        note_dict[note_obj.offset].append(n)
                else:
                    note_dict[note_obj.offset].append(note_obj)
            
            # For each time offset, sort notes by pitch (highest to lowest)
            voice_assignments = defaultdict(stream.Stream)
            
            for offset, notes_at_offset in sorted(note_dict.items()):
                # Sort notes by pitch, highest first
                sorted_notes = sorted(notes_at_offset, key=lambda n: n.pitch.midi if hasattr(n, 'pitch') else 0, reverse=True)
                
                # Assign notes to voices (0 is highest voice/melody)
                for voice_idx, note_obj in enumerate(sorted_notes):
                    voice_stream = voice_assignments[voice_idx]
                    voice_stream.insert(offset, note_obj)
                    
            # Convert to list of parts
            parts = [voice_assignments[i] for i in sorted(voice_assignments.keys())]
        
        # Create output files for each voice
        output_files = []
        
        # Sort parts by their average pitch, highest first
        def get_avg_pitch(part):
            notes = [n for n in part.flat.notes if hasattr(n, 'pitch')]
            if not notes:
                return 0
            return sum(n.pitch.midi for n in notes) / len(notes)
        
        sorted_parts = sorted(parts, key=get_avg_pitch, reverse=True)
        
        # Save each voice to a separate file
        for i, part in enumerate(sorted_parts):
            if i == 0:
                voice_name = "melody"
                part.insert(0, instrument.Piano())
            else:
                voice_name = f"harmony{i}"
                part.insert(0, instrument.Piano())
            
            output_path = os.path.join(midi_output_dir, f"{midi_name}_{voice_name}.mid")
            part.write('midi', fp=output_path)
            output_files.append(output_path)
            print(f"  Created {voice_name} file: {os.path.basename(output_path)}")
        
        return output_files
    
    except Exception as e:
        print(f"Error processing {midi_file}: {str(e)}")
        return []

def process_midi_folder(input_folder, output_folder):
    """
    Process all MIDI files in a folder to extract individual voices.
    
    Args:
        input_folder (str): Path to the folder containing MIDI files
        output_folder (str): Path to save extracted voices
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all MIDI files
    midi_files = glob.glob(os.path.join(input_folder, "*.mid")) + \
                glob.glob(os.path.join(input_folder, "*.midi"))
    
    if not midi_files:
        print(f"No MIDI files found in {input_folder}")
        return
    
    print(f"Found {len(midi_files)} MIDI files to process")
    
    # Process each MIDI file
    all_output_files = []
    for midi_file in midi_files:
        output_files = extract_voices(midi_file, output_folder)
        all_output_files.extend(output_files)
    
    print(f"Processing complete. Created {len(all_output_files)} voice files in {output_folder}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract individual voices from polyphonic MIDI files")
    parser.add_argument("--input", "-i", default="HarmDataset", 
                        help="Input folder containing MIDI files")
    parser.add_argument("--output", "-o", default="ProcessedDataset",
                        help="Output folder for extracted voices")
    
    args = parser.parse_args()
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, args.input)
    output_folder = os.path.join(current_dir, args.output)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Process all MIDI files
    process_midi_folder(input_folder, output_folder)