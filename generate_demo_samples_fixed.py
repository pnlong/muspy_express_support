# README
# Phillip Long
# January 25, 2024

# Generate music samples for each model and save as before/after audio files.

# python /home/pnlong/muspy_express/generate_demo_samples_all.py


# IMPORTS
##################################################

import argparse
import logging
import pprint
import sys
from os.path import exists, dirname, basename
from os import makedirs
from typing import Callable, Tuple
import multiprocessing
from shutil import rmtree
import subprocess
import contextlib
import os
import warnings

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

# Suppress x_transformers FutureWarning about deprecated torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, module="x_transformers")

# Suppress musicxml MIDI channel warnings
warnings.filterwarnings("ignore", message="WARNING: we are out of midi channels! help!", module="musicxml.m21ToXml")

from read_mscz.music import MusicExpress
import dataset
import music_x_transformers
import representation
import encode
import decode
import utils
import train
from evaluate_baseline import pad, unpad_prefix # for padding batches

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/deepfreeze/pnlong/muspy_express/experiments/metrical"
PATHS = f"{DATA_DIR}/test.txt"
OUTPUT_DIR = "/deepfreeze/pnlong/muspy_express/demo_samples"
EVAL_STEM = "eval"

DEFAULT_MAX_PREFIX_LEN = 50

CONDITIONAL_TYPES = train.MASKS
GENERATION_TYPES = train.MASKS
EVAL_TYPES = ["joint",] + [f"conditional_{conditional_type}_{generation_type}" for conditional_type in CONDITIONAL_TYPES for generation_type in GENERATION_TYPES]

LINE = "-" * 60

##################################################


# ARGUMENTS
##################################################
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", default = PATHS, type = str, help = ".txt file with absolute filepaths to testing dataset.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-ns", "--n_samples", type = int, default = 10, help = "Number of samples to evaluate")
    # model
    parser.add_argument("--seq_len", default = train.DEFAULT_MAX_SEQ_LEN, type = int, help = "Sequence length to generate")
    parser.add_argument("--temperature", nargs = "+", default = 1.0, type = float, help = "Sampling temperature (default: 1.0)")
    parser.add_argument("--filter", nargs = "+", default = "top_k", type = str, help = "Sampling filter (default: 'top_k')")
    parser.add_argument("--filter_threshold", nargs = "+", default = 0.9, type = float, help = "Sampling filter threshold (default: 0.9)")
    parser.add_argument("--prefix_len", default = DEFAULT_MAX_PREFIX_LEN, type = int, help = "Number of notes in prefix sequence for generation")
    parser.add_argument("--expr_prefix_scale", default = 0.1, type = float, help = "Scale of prefix_len to use for expressive-prefix budget in econditional models")
    parser.add_argument("--prefix_include_expressive_windowed", default = True, type = bool, help = "For non-conditional prefix/anticipation, include expressive up to nth note boundary")
    # others
    parser.add_argument("-bs", "--batch_size", default = 8, type = int, help = "Batch size")
    parser.add_argument("-g", "--gpu", type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 4, type = int, help = "Number of jobs")
    parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
    parser.add_argument("--debug", action = "store_true", help = "Enable debug logging for troubleshooting.")
    return parser.parse_args(args = args, namespace = namespace)
##################################################


# HELPER FUNCTIONS
##################################################

@contextlib.contextmanager
def suppress_fluidsynth_warnings():
    """Context manager to suppress fluidsynth warnings during audio synthesis."""
    with open(os.devnull, 'w') as devnull:
        # Temporarily redirect stderr to suppress fluidsynth warnings
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

def write_music_files_silenced(
    music: MusicExpress,
    output_dir: str,
    prefix: str,
    skip_existing: bool = True
) -> bool:
    """
    Write music object to WAV, MIDI, and MusicXML files with suppressed fluidsynth warnings.
    
    Parameters
    ----------
    music : MusicExpress
        Music object to write.
    output_dir : str
        Directory to write files to.
    prefix : str
        Prefix for filenames (e.g., "before" or "after").
    skip_existing : bool
        Whether to skip if files already exist.
        
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Define file paths
        wav_filepath = f"{output_dir}/{prefix}.wav"
        midi_filepath = f"{output_dir}/{prefix}.mid"
        mxl_filepath = f"{output_dir}/{prefix}.mxl"
        
        # Check if all files exist and skip if requested
        required_files = [wav_filepath, midi_filepath]
        if skip_existing and all(exists(fp) for fp in required_files):
            return True
        
        # Debug: Validate music object before writing
        logging.debug(f"Writing music files for {output_dir}/{prefix}")
        logging.debug(f"  Music object: {type(music)}")
        logging.debug(f"  Number of tracks: {len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}")
        logging.debug(f"  Resolution: {getattr(music, 'resolution', 'N/A')}")
        logging.debug(f"  Song length: {getattr(music, 'song_length', 'N/A')}")
        
        # Check for potential issues
        if hasattr(music, 'tracks') and len(music.tracks) == 0:
            logging.warning(f"Music object has no tracks for {output_dir}/{prefix}")
        
        # Check for empty music objects
        total_notes = 0
        if hasattr(music, 'tracks'):
            total_notes = sum(len(track.notes) if hasattr(track, 'notes') else 0 for track in music.tracks)
        
        if len(music.tracks) == 0 or total_notes == 0:
            logging.warning(f"Empty music object detected for {output_dir}/{prefix}, skipping file writing")
            return False
        
        # Count notes per track for debugging
        if hasattr(music, 'tracks'):
            for i, track in enumerate(music.tracks):
                note_count = len(track.notes) if hasattr(track, 'notes') else 0
                logging.debug(f"  Track {i}: {note_count} notes")
                if note_count == 0:
                    logging.warning(f"Track {i} has no notes for {output_dir}/{prefix}")
        
        # Check for valid tempo before writing MusicXML
        musicxml_skip = False
        if hasattr(music, 'tempos') and len(music.tempos) > 0:
            for tempo in music.tempos:
                if hasattr(tempo, 'qpm') and (tempo.qpm == 0 or tempo.qpm is None):
                    logging.warning(f"Invalid tempo detected: {tempo.qpm}, will skip MusicXML for {output_dir}/{prefix}")
                    musicxml_skip = True
                    break
        
        # Write files with individual error handling and suppressed fluidsynth warnings
        try:
            logging.debug(f"  Writing audio file: {wav_filepath}")
            with suppress_fluidsynth_warnings():
                music.write(wav_filepath, kind = "audio")
            logging.debug(f"  Successfully wrote audio file")
        except Exception as e:
            logging.error(f"Failed to write audio file {wav_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            raise
        
        try:
            logging.debug(f"  Writing MIDI file: {midi_filepath}")
            music.write(midi_filepath, kind = "midi")
            logging.debug(f"  Successfully wrote MIDI file")
        except Exception as e:
            logging.error(f"Failed to write MIDI file {midi_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            raise
        
        try:
            if musicxml_skip:
                logging.debug(f"  Skipping MusicXML file due to invalid tempo: {mxl_filepath}")
            else:
                logging.debug(f"  Writing MusicXML file: {mxl_filepath}")
                music.write(mxl_filepath, kind = "musicxml")
                logging.debug(f"  Successfully wrote MusicXML file")
        except Exception as e:
            logging.error(f"Failed to write MusicXML file {mxl_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            # Don't raise for MusicXML errors, continue with other files
            logging.warning(f"  Continuing without MusicXML file")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to write music files for {output_dir}/{prefix}: {e}")
        logging.error(f"  Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"  Traceback: {traceback.format_exc()}")
        return False

def analyze_sequence_tokens(sequence_data: np.array, encoding: dict, model_name: str, stem: str, sequence_type: str) -> dict:
    """
    Analyze and log sequence token contents for debugging.
    
    Parameters
    ----------
    sequence_data : np.array
        The sequence data to analyze.
    encoding : dict
        Encoding configuration.
    model_name : str
        Name of the model.
    stem : str
        Stem identifier for this sample.
    sequence_type : str
        Type of sequence ("prefix", "generated", "full").
        
    Returns
    -------
    dict
        Dictionary with token counts.
    """
    try:
        logging.debug(f"=== {sequence_type.upper()} ANALYSIS for {model_name} sample {stem} ===")
        logging.debug(f"  {sequence_type.capitalize()} shape: {sequence_data.shape}")
        logging.debug(f"  {sequence_type.capitalize()} dtype: {sequence_data.dtype}")
        
        # Get token mappings
        type_code_map = encoding["type_code_map"]
        sos_token = type_code_map["start-of-song"]
        eos_token = type_code_map["end-of-song"]
        note_token = type_code_map["note"]
        grace_note_token = type_code_map["grace-note"]
        expressive_token = type_code_map[representation.EXPRESSIVE_FEATURE_TYPE_STRING]
        
        # Analyze token types
        if len(sequence_data.shape) == 1:
            # Unidimensional case
            type_dim = encoding["unidimensional_encoding_order"].index("type")
            type_tokens = sequence_data[type_dim::len(encoding["dimensions"])]
        else:
            # Multidimensional case
            type_dim = encoding["dimensions"].index("type")
            type_tokens = sequence_data[:, type_dim]
        
        # Count different token types
        sos_count = np.sum(type_tokens == sos_token)
        eos_count = np.sum(type_tokens == eos_token)
        note_count = np.sum((type_tokens == note_token) | (type_tokens == grace_note_token))
        expressive_count = np.sum(type_tokens == expressive_token)
        other_count = len(type_tokens) - sos_count - eos_count - note_count - expressive_count
        
        token_counts = {
            "sos": int(sos_count),
            "eos": int(eos_count),
            "notes": int(note_count),
            "expressive": int(expressive_count),
            "other": int(other_count),
            "total_events": len(type_tokens)
        }
        
        logging.debug(f"  Token counts:")
        logging.debug(f"    SOS: {token_counts['sos']}")
        logging.debug(f"    EOS: {token_counts['eos']}")
        logging.debug(f"    Notes: {token_counts['notes']}")
        logging.debug(f"    Expressive: {token_counts['expressive']}")
        logging.debug(f"    Other: {token_counts['other']}")
        logging.debug(f"    Total events: {token_counts['total_events']}")
        
        # Show first few tokens
        logging.debug(f"  First 10 type tokens: {type_tokens[:10] if len(type_tokens) > 10 else type_tokens}")
        
        # Check if sequence is meaningful
        if note_count > 0:
            logging.debug(f"  ✓ {sequence_type.capitalize()} contains {note_count} notes")
        if expressive_count > 0:
            logging.debug(f"  ✓ {sequence_type.capitalize()} contains {expressive_count} expressive features")
        if note_count == 0 and expressive_count == 0:
            logging.debug(f"  ⚠ {sequence_type.capitalize()} appears to be empty/minimal (only SOS/EOS)")
        
        logging.debug(f"=== END {sequence_type.upper()} ANALYSIS ===")
        
        return token_counts
        
    except Exception as e:
        logging.warning(f"Failed to analyze {sequence_type} sequence for {model_name} sample {stem}: {e}")
        return {"sos": 0, "eos": 0, "notes": 0, "expressive": 0, "other": 0, "total_events": 0}

def analyze_instruments(music: MusicExpress, model_name: str, stem: str, sequence_type: str) -> dict:
    """
    Analyze instrument usage in a music object.
    
    Parameters
    ----------
    music : MusicExpress
        Music object to analyze.
    model_name : str
        Name of the model.
    stem : str
        Stem identifier for this sample.
    sequence_type : str
        Type of sequence ("prefix", "generated", "full").
        
    Returns
    -------
    dict
        Dictionary with instrument analysis results.
    """
    try:
        instrument_info = {
            "total_tracks": len(music.tracks),
            "unique_programs": set(),
            "drum_tracks": 0,
            "non_drum_tracks": 0,
            "program_counts": {},
            "track_details": []
        }
        
        for i, track in enumerate(music.tracks):
            program = track.program
            is_drum = track.is_drum
            note_count = len(track.notes) if hasattr(track, 'notes') else 0
            
            instrument_info["unique_programs"].add(program)
            instrument_info["program_counts"][program] = instrument_info["program_counts"].get(program, 0) + 1
            
            if is_drum:
                instrument_info["drum_tracks"] += 1
            else:
                instrument_info["non_drum_tracks"] += 1
            
            instrument_info["track_details"].append({
                "track_index": i,
                "program": program,
                "is_drum": is_drum,
                "note_count": note_count,
                "name": getattr(track, 'name', f"Track {i}")
            })
        
        instrument_info["unique_programs"] = len(instrument_info["unique_programs"])
        
        # Log instrument analysis
        logging.debug(f"=== INSTRUMENT ANALYSIS for {model_name} sample {stem} ({sequence_type}) ===")
        logging.debug(f"  Total tracks: {instrument_info['total_tracks']}")
        logging.debug(f"  Unique programs: {instrument_info['unique_programs']}")
        logging.debug(f"  Drum tracks: {instrument_info['drum_tracks']}")
        logging.debug(f"  Non-drum tracks: {instrument_info['non_drum_tracks']}")
        
        # Check for MIDI channel issues
        # MIDI has 16 channels: 0-8 and 10-15 (channel 9 is reserved for drums)
        # So we have 15 channels for non-drum instruments
        available_channels = 15
        if instrument_info["non_drum_tracks"] > available_channels:
            logging.warning(f"  ⚠ MIDI CHANNEL EXHAUSTION: {instrument_info['non_drum_tracks']} non-drum tracks > {available_channels} available channels")
            logging.warning(f"     This will cause channel conflicts and fluidsynth errors")
        else:
            logging.debug(f"  ✓ Channel usage OK: {instrument_info['non_drum_tracks']} non-drum tracks <= {available_channels} channels")
        
        # Log program usage
        for program, count in sorted(instrument_info["program_counts"].items()):
            logging.debug(f"    Program {program}: {count} track(s)")
        
        logging.debug(f"=== END INSTRUMENT ANALYSIS ===")
        
        return instrument_info
        
    except Exception as e:
        logging.warning(f"Failed to analyze instruments for {model_name} sample {stem} ({sequence_type}): {e}")
        return {"total_tracks": 0, "unique_programs": 0, "drum_tracks": 0, "non_drum_tracks": 0, "program_counts": {}, "track_details": []}

def analyze_generation_results(prefix_data: np.array, generated_data: np.array, encoding: dict, model_name: str, stem: str) -> None:
    """
    Analyze generation results by comparing prefix and full generated sequence.
    
    Parameters
    ----------
    prefix_data : np.array
        The prefix sequence data.
    generated_data : np.array
        The full generated sequence (prefix + generated).
    encoding : dict
        Encoding configuration.
    model_name : str
        Name of the model.
    stem : str
        Stem identifier for this sample.
    """
    try:
        # Analyze prefix and full sequence
        prefix_counts = analyze_sequence_tokens(prefix_data, encoding, model_name, stem, "prefix")
        full_counts = analyze_sequence_tokens(generated_data, encoding, model_name, stem, "full")
        
        # Calculate what was newly generated (this is the actual generation, not including prefix)
        new_notes = full_counts["notes"] - prefix_counts["notes"]
        new_expressive = full_counts["expressive"] - prefix_counts["expressive"]
        new_events = full_counts["total_events"] - prefix_counts["total_events"]
        
        # Log generation summary
        logging.info(f"=== GENERATION SUMMARY for {model_name} sample {stem} ===")
        logging.info(f"  Prefix: {prefix_counts['notes']} notes, {prefix_counts['expressive']} expressive, {prefix_counts['total_events']} total events")
        logging.info(f"  Full: {full_counts['notes']} notes, {full_counts['expressive']} expressive, {full_counts['total_events']} total events")
        logging.info(f"  NEWLY GENERATED: {new_notes} notes, {new_expressive} expressive, {new_events} total events")
        
        # Check model-specific expectations
        is_conditional = "conditional" in model_name and "econditional" not in model_name
        is_econditional = "econditional" in model_name
        is_joint = "joint" in model_name or (not is_conditional and not is_econditional)
        
        if is_conditional:
            # Conditional models should generate notes but not expressive features
            if new_expressive > 0:
                logging.warning(f"  ⚠ Conditional model generated {new_expressive} expressive features (should be 0)")
            else:
                logging.info(f"  ✓ Conditional model correctly generated 0 expressive features")
                
        elif is_econditional:
            # Econditional models should generate expressive features but not notes
            if new_notes > 0:
                logging.warning(f"  ⚠ Econditional model generated {new_notes} notes (should be 0)")
            else:
                logging.info(f"  ✓ Econditional model correctly generated 0 notes")
                
        elif is_joint:
            # Joint models can generate both
            logging.info(f"  ✓ Joint model generated both notes and expressive features")
        
        logging.info(f"=== END GENERATION SUMMARY ===")
        
    except Exception as e:
        logging.warning(f"Failed to analyze generation results for {model_name} sample {stem}: {e}")

def determine_model_config(model_name: str) -> str:
    """
    Determine the evaluation subdirectory for a model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to analyze.
        
    Returns
    -------
    str
        Eval subdirectory name for this model.
    """
    
    # check model type - must check econditional before conditional since it contains 'conditional'
    if "econditional" in model_name:
        return "conditional_note_total"
    elif "conditional" in model_name:
        return "conditional_expressive_total"
    else:
        return "joint"


def _is_note(type_id: int, note_token: int, grace_note_token: int) -> bool:
    return (type_id == note_token) or (type_id == grace_note_token)


def _is_expressive(type_id: int, expressive_token: int) -> bool:
    return type_id == expressive_token


def _event_positions(seq_len: int, type_dim: int, n_tokens_per_event: int) -> range:
    return range(type_dim, seq_len, n_tokens_per_event)


def _find_first_sos_index(type_field_seq: torch.Tensor, sos_token: int) -> int:
    # type_field_seq is 1D over event positions
    matches = (type_field_seq == sos_token).nonzero(as_tuple = True)[0]
    return int(matches[0].item()) if matches.numel() > 0 else 0


def _gather_indices_by_predicate(type_field_seq: torch.Tensor, start_event_idx: int, predicate_fn) -> torch.Tensor:
    # Returns 1D tensor of event indices (relative to event positions list) that satisfy predicate
    if start_event_idx > 0:
        type_field_seq = type_field_seq[start_event_idx:]
        base = start_event_idx
    else:
        base = 0
    mask = torch.tensor([bool(predicate_fn(int(t.item()))) for t in type_field_seq], device = type_field_seq.device)
    idx = mask.nonzero(as_tuple = True)[0]
    return idx + base


def _slice_events_by_indices(seq: torch.Tensor, event_pos_indices: torch.Tensor, n_tokens_per_event: int, unidimensional: bool) -> torch.Tensor:
    # Concatenate token rows for each selected event index
    if event_pos_indices.numel() == 0:
        return seq.new_zeros((0, seq.shape[-1])) if not unidimensional else seq.new_zeros((0,))
    slices = []
    for pos in event_pos_indices.tolist():
        start = pos
        end = pos + n_tokens_per_event
        if unidimensional:
            slices.append(seq[start:end])
        else:
            slices.append(seq[start:end, :])
    if unidimensional:
        return torch.cat(slices, dim = 0)
    return torch.cat(slices, dim = 0)


def _merge_and_sort_by_event_index(indices_groups: list) -> torch.Tensor:
    # indices_groups: list of 1D LongTensors of event indices
    if not indices_groups:
        return torch.empty(0, dtype = torch.long)
    merged = torch.cat([g for g in indices_groups if g.numel() > 0], dim = 0)
    if merged.numel() == 0:
        return merged
    return torch.sort(merged).values


def write_music_files(
    music: MusicExpress,
    output_dir: str,
    prefix: str,
    skip_existing: bool = True
) -> bool:
    """
    Write music object to WAV, MIDI, and MusicXML files.
    
    Parameters
    ----------
    music : MusicExpress
        Music object to write.
    output_dir : str
        Directory to write files to.
    prefix : str
        Prefix for filenames (e.g., "before" or "after").
    skip_existing : bool
        Whether to skip if files already exist.
        
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Define file paths
        wav_filepath = f"{output_dir}/{prefix}.wav"
        midi_filepath = f"{output_dir}/{prefix}.mid"
        mxl_filepath = f"{output_dir}/{prefix}.mxl"
        
        # Check if all files exist and skip if requested
        # Note: We check for MusicXML separately since it might be skipped due to tempo issues
        required_files = [wav_filepath, midi_filepath]
        if skip_existing and all(exists(fp) for fp in required_files):
            # If MusicXML exists too, that's a bonus, but not required
            return True
        
        # Debug: Validate music object before writing
        logging.debug(f"Writing music files for {output_dir}/{prefix}")
        logging.debug(f"  Music object: {type(music)}")
        logging.debug(f"  Number of tracks: {len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}")
        logging.debug(f"  Resolution: {getattr(music, 'resolution', 'N/A')}")
        logging.debug(f"  Song length: {getattr(music, 'song_length', 'N/A')}")
        
        # Check for potential issues
        if hasattr(music, 'tracks') and len(music.tracks) == 0:
            logging.warning(f"Music object has no tracks for {output_dir}/{prefix}")
        
        # Check for empty music objects
        total_notes = 0
        if hasattr(music, 'tracks'):
            total_notes = sum(len(track.notes) if hasattr(track, 'notes') else 0 for track in music.tracks)
        
        if len(music.tracks) == 0 or total_notes == 0:
            logging.warning(f"Empty music object detected for {output_dir}/{prefix}, skipping file writing")
            return False
        
        # Count notes per track for debugging
        if hasattr(music, 'tracks'):
            for i, track in enumerate(music.tracks):
                note_count = len(track.notes) if hasattr(track, 'notes') else 0
                logging.debug(f"  Track {i}: {note_count} notes")
                if note_count == 0:
                    logging.warning(f"Track {i} has no notes for {output_dir}/{prefix}")
        
        # Check for valid tempo before writing MusicXML
        musicxml_skip = False
        if hasattr(music, 'tempos') and len(music.tempos) > 0:
            for tempo in music.tempos:
                if hasattr(tempo, 'qpm') and (tempo.qpm == 0 or tempo.qpm is None):
                    logging.warning(f"Invalid tempo detected: {tempo.qpm}, will skip MusicXML for {output_dir}/{prefix}")
                    musicxml_skip = True
                    break
        
        # Write files with individual error handling
        try:
            logging.debug(f"  Writing audio file: {wav_filepath}")
            music.write(wav_filepath, kind = "audio")
            logging.debug(f"  Successfully wrote audio file")
        except Exception as e:
            logging.error(f"Failed to write audio file {wav_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            raise
        
        try:
            logging.debug(f"  Writing MIDI file: {midi_filepath}")
            music.write(midi_filepath, kind = "midi")
            logging.debug(f"  Successfully wrote MIDI file")
        except Exception as e:
            logging.error(f"Failed to write MIDI file {midi_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            raise
        
        try:
            if musicxml_skip:
                logging.debug(f"  Skipping MusicXML file due to invalid tempo: {mxl_filepath}")
            else:
                logging.debug(f"  Writing MusicXML file: {mxl_filepath}")
                music.write(mxl_filepath, kind = "musicxml")
                logging.debug(f"  Successfully wrote MusicXML file")
        except Exception as e:
            logging.error(f"Failed to write MusicXML file {mxl_filepath}: {e}")
            logging.error(f"  Music object details: tracks={len(music.tracks) if hasattr(music, 'tracks') else 'N/A'}, resolution={getattr(music, 'resolution', 'N/A')}")
            # Don't raise for MusicXML errors, continue with other files
            logging.warning(f"  Continuing without MusicXML file")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to write music files for {output_dir}/{prefix}: {e}")
        logging.error(f"  Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"  Traceback: {traceback.format_exc()}")
        return False

def process_single_sample(
    generated_data: np.array,
    prefix_data: np.array,
    stem: str,
    eval_type: str,
    output_base_dir: str,
    encoding: dict,
    sos_token: int,
    skip_existing: bool = True,
    unidimensional_decoding_function: Callable = None
) -> Tuple[str, bool]:
    """
    Process a single generated sample into before/after audio files.
    
    Parameters
    ----------
    generated_data : np.array
        The generated sequence (prefix + generated).
    prefix_data : np.array
        The prefix sequence.
    stem : str
        Stem for the output directory name.
    eval_type : str
        Type of evaluation (joint, conditional_*, etc.).
    output_base_dir : str
        Base output directory.
    encoding : dict
        Encoding configuration.
    sos_token : int
        Start-of-song token.
    skip_existing : bool
        Whether to skip existing files.
    unidimensional_decoding_function : Callable
        Function for unidimensional decoding.
        
    Returns
    -------
    Tuple[str, bool]
        Tuple containing (output_dir, success).
    """
    try:
        # Create output directory
        output_dir = f"{output_base_dir}/{stem}"
        
        # Check if all expected files exist and skip if requested
        expected_files = [f"{output_dir}/after.wav", f"{output_dir}/after.mid", f"{output_dir}/after.mxl"]
        if skip_existing and all(exists(fp) for fp in expected_files):
            return output_dir, True
        
        # Create output directory
        makedirs(output_dir, exist_ok = True)
        
        # Debug: Log sequence information before decoding
        logging.debug(f"Decoding generated sequence for {stem}")
        logging.debug(f"  Generated data shape: {generated_data.shape}")
        logging.debug(f"  Generated data type: {type(generated_data)}")
        logging.debug(f"  First 10 elements: {generated_data[:10] if len(generated_data) > 10 else generated_data}")
        
        # Decode the full generated sequence
        try:
            generated_music = decode.decode(
                codes = generated_data,
                encoding = encoding,
                infer_metrical_time = True,
                unidimensional_decoding_function = unidimensional_decoding_function
            )
            logging.debug(f"  Successfully decoded generated sequence")
            
            # Analyze instruments in the generated music
            analyze_instruments(
                music = generated_music,
                model_name = model_name,
                stem = stem,
                sequence_type = "full"
            )
        except Exception as e:
            logging.error(f"Failed to decode generated sequence for {stem}: {e}")
            logging.error(f"  Generated data shape: {generated_data.shape}")
            logging.error(f"  Generated data sample: {generated_data[:20] if len(generated_data) > 20 else generated_data}")
            raise
        
        # Write "after" files (full sequence)
        success_after = write_music_files_silenced(
            music = generated_music,
            output_dir = output_dir,
            prefix = "after",
            skip_existing = skip_existing
        )
        
        if not success_after:
            return output_dir, False
        
        # Write "before" files for all models since they all have prefixes
        # All models now use meaningful prefixes, not just SOS tokens
        if True:  # Always generate before files
            # Debug: Log prefix sequence information before decoding
            logging.debug(f"Decoding prefix sequence for {stem}")
            logging.debug(f"  Prefix data shape: {prefix_data.shape}")
            logging.debug(f"  Prefix data type: {type(prefix_data)}")
            logging.debug(f"  First 10 elements: {prefix_data[:10] if len(prefix_data) > 10 else prefix_data}")
            
            # Decode the prefix sequence
            try:
                prefix_music = decode.decode(
                    codes = prefix_data,
                    encoding = encoding,
                    infer_metrical_time = True,
                    unidimensional_decoding_function = unidimensional_decoding_function
                )
                logging.debug(f"  Successfully decoded prefix sequence")
                
                # Analyze instruments in the prefix music
                analyze_instruments(
                    music = prefix_music,
                    model_name = model_name,
                    stem = stem,
                    sequence_type = "prefix"
                )
            except Exception as e:
                logging.error(f"Failed to decode prefix sequence for {stem}: {e}")
                logging.error(f"  Prefix data shape: {prefix_data.shape}")
                logging.error(f"  Prefix data sample: {prefix_data[:20] if len(prefix_data) > 20 else prefix_data}")
                raise
            
            # Write "before" files (prefix only)
            success_before = write_music_files_silenced(
                music = prefix_music,
                output_dir = output_dir,
                prefix = "before",
                skip_existing = skip_existing
            )
            
            if not success_before:
                return output_dir, False
        
        return output_dir, True
        
    except Exception as e:
        logging.error(f"Failed to process sample {stem} for {eval_type}: {e}")
        return "", False

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # get directories
    experiment_dir = dirname(args.paths)
    experiment_type = basename(experiment_dir)  # Default, could be determined from args.paths
    output_base_dir = f"{args.output_dir}/{experiment_type}"

    # create output directory structure
    # Determine experiment type from paths or use default
    removed_output_dir_because_reset = False
    if args.reset and exists(output_base_dir):
        removed_output_dir_because_reset = True
        rmtree(output_base_dir)
    if not exists(output_base_dir):
        makedirs(output_base_dir)

    # set up the logger
    stream_handler = logging.StreamHandler(stream = sys.stdout)
    stream_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    file_handler = logging.FileHandler(filename = f"{output_base_dir}/generation.log", mode = "w")
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(level = logging.DEBUG, format = "%(message)s", handlers = [stream_handler, file_handler])
    
    if args.debug:
        logging.info("Debug logging enabled - this will provide detailed information about processing steps")
    
    # make note of removing output directory because --reset was specified
    if removed_output_dir_because_reset:
        logging.info(f"Removing existing output directory because --reset was specified: {output_base_dir}")
    
    # Define the 7 model configurations
    def determine_model_order(name: str) -> float:
        val = 0.5 if "anticipation" in name else 0.0
        if "baseline" in name:
            val += 0
        elif "econditional" in name:
            val += 3
        elif "conditional" in name:
            val += 2
        else:
            val += 1
        return val

    with open(f"{experiment_dir}/models/models.txt", "r") as f:
        MODEL_CONFIGS = [line.strip() for line in f.readlines() if line.strip()]
        MODEL_CONFIGS = sorted(MODEL_CONFIGS, key = determine_model_order)

    # load the encoding
    encoding_filepath = f"{experiment_dir}/encoding.json"
    encoding = representation.load_encoding(filepath = encoding_filepath) if exists(encoding_filepath) else representation.get_encoding()

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{output_base_dir}/generation_args.json"
    logging.info(f"Saved arguments to {args_output_filepath}")
    utils.save_args(filepath = args_output_filepath, args = args)
    del args_output_filepath

    ##################################################


    # SETUP DEVICE
    ##################################################

    # get the specified device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    logging.info(f"Using device: {device}")
    
    # get max_seq_len from args
    max_seq_len = args.seq_len

    ##################################################


    # GENERATE SAMPLES FOR EACH MODEL
    ##################################################

    # iterate over each model configuration
    logging.info(LINE)
    
    # Track success/failure counts for each model
    model_results = {}
    
    for model_name in MODEL_CONFIGS:
        logging.info(f"Processing model: {model_name}")
        
        # determine the eval type for this model
        eval_type = determine_model_config(model_name)
        logging.info(f"  Using eval type: {eval_type}")
        
        # Initialize counters for this model
        model_results[model_name] = {
            "eval_type": eval_type,
            "total_processed": 0,
            "total_successful": 0
        }
        
        # create model output directory
        model_output_dir = f"{output_base_dir}/{model_name}"
        if not exists(model_output_dir):
            makedirs(model_output_dir)
        
        # load model-specific training arguments
        model_train_args_filepath = f"{experiment_dir}/models/{model_name}/train_args.json"
        if not exists(model_train_args_filepath):
            logging.warning(f"Training arguments not found for {model_name}, skipping...")
            model_results[model_name]["total_processed"] = 0
            model_results[model_name]["total_successful"] = 0
            continue
            
        logging.debug(f"Loading training arguments from: {model_train_args_filepath}")
        model_train_args = utils.load_json(filepath = model_train_args_filepath)
        
        # create model-specific dataset
        model_conditioning = model_train_args["conditioning"]
        model_unidimensional = model_train_args.get("unidimensional", False)
        model_test_dataset = dataset.MusicDataset(
            paths = args.paths, 
            encoding = encoding, 
            conditioning = model_conditioning, 
            max_seq_len = max_seq_len, 
            use_augmentation = False, 
            is_baseline = ("baseline" in model_name), 
            unidimensional = model_unidimensional, 
            for_generation = True
        )
        
        # create model-specific model
        model_use_absolute_time = encoding["use_absolute_time"]
        model_model = music_x_transformers.MusicXTransformer(
            dim = model_train_args["dim"],
            encoding = encoding,
            depth = model_train_args["layers"],
            heads = model_train_args["heads"],
            max_seq_len = max_seq_len,
            max_temporal = encoding["max_" + ("time" if model_use_absolute_time else "beat")],
            rotary_pos_emb = model_train_args["rel_pos_emb"],
            use_abs_pos_emb = model_train_args["abs_pos_emb"],
            emb_dropout = model_train_args["dropout"],
            attn_dropout = model_train_args["dropout"],
            ff_dropout = model_train_args["dropout"],
            unidimensional = model_unidimensional,
        ).to(device)
        
        # load model checkpoint
        model_checkpoint_dir = f"{experiment_dir}/models/{model_name}/checkpoints"
        model_checkpoint_filepath = f"{model_checkpoint_dir}/best_model.{train.PARTITIONS[1]}.pth"
        if not exists(model_checkpoint_filepath):
            logging.warning(f"Checkpoint not found for {model_name}, skipping...")
            model_results[model_name]["total_processed"] = 0
            model_results[model_name]["total_successful"] = 0
            continue
            
        model_state_dict = torch.load(f = model_checkpoint_filepath, map_location = device, weights_only = True)
        model_model.load_state_dict(state_dict = model_state_dict)
        model_model.eval()
        
        # get model-specific tokens and settings
        if model_unidimensional:
            model_unidimensional_encoding_order = encoding["unidimensional_encoding_order"]
        model_type_dim = (model_unidimensional_encoding_order if model_unidimensional else encoding["dimensions"]).index("type")
        model_sos = encoding["type_code_map"]["start-of-song"]
        model_eos = encoding["type_code_map"]["end-of-song"]
        model_note_token, model_grace_note_token = encoding["type_code_map"]["note"], encoding["type_code_map"]["grace-note"]
        model_expressive_feature_token = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]
        model_conditional_on_controls = (bool(model_train_args.get("conditional", False)) or bool(model_train_args.get("econditional", False)))
        model_notes_are_controls = bool(model_train_args.get("econditional", False))
        # For prefix extraction, we always want to count note tokens as events, regardless of controls
        # This ensures we extract meaningful musical prefixes with actual notes
        model_event_tokens = torch.tensor(data = (model_note_token, model_grace_note_token), device = device)
        model_is_anticipation = (model_conditioning == encode.CONDITIONINGS[-1])
        model_sigma = model_train_args["sigma"]
        model_unidimensional_encoding_function, model_unidimensional_decoding_function = representation.get_unidimensional_coding_functions(encoding = encoding)
        model_get_type_field = lambda prefix_conditional: prefix_conditional[model_type_dim::model_model.decoder.net.n_tokens_per_event] if model_unidimensional else prefix_conditional[:, model_type_dim]
        
        # create data loader for this model
        model_test_data_loader = torch.utils.data.DataLoader(
            dataset = model_test_dataset, 
            num_workers = args.jobs, 
            collate_fn = model_test_dataset.collate, 
            batch_size = args.batch_size, 
            shuffle = False
        )
        model_test_iter = iter(model_test_data_loader)
        chunk_size = int(args.batch_size / 2)

        # iterate over the dataset for this model
        with torch.no_grad():
            n_iterations = (int((args.n_samples - 1) / args.batch_size) + 1) if args.n_samples is not None else len(model_test_data_loader)
            total_samples_processed = 0
            total_successful_samples = 0
            
            # Create custom progress bar for samples
            total_samples_to_process = args.n_samples if args.n_samples is not None else len(model_test_dataset)
            pbar = tqdm(total = total_samples_to_process, desc = f"Generating samples for {model_name}")
            
            for i in range(n_iterations):
                
                # get new batch
                batch = next(model_test_iter)
                stem = i
                if (args.n_samples is not None) and (i == n_iterations - 1): # if last iteration
                    n_samples = args.n_samples % args.batch_size
                    if (n_samples == 0):
                        n_samples = args.batch_size
                    batch["seq"] = batch["seq"][:n_samples]
                    batch["mask"] = batch["mask"][:n_samples]
                    batch["seq_len"] = batch["seq_len"][:n_samples]
                    batch["path"] = batch["path"][:n_samples]

                # DETERMINE PREFIX SEQUENCE (model-specific)
                ##################################################

                # Helpers for type access per sample
                def get_type_events(sample_seq: torch.Tensor) -> torch.Tensor:
                    if model_unidimensional:
                        # Extract type dimension across event positions
                        type_stream = []
                        for j in range(model_type_dim, sample_seq.shape[0], model_model.decoder.net.n_tokens_per_event):
                            type_stream.append(sample_seq[j])
                        return torch.stack(type_stream, dim = 0)
                    else:
                        return sample_seq[model_type_dim::model_model.decoder.net.n_tokens_per_event, model_type_dim]

                is_prefix_model = ("prefix" in model_name)
                is_anticipation_model = model_is_anticipation
                is_conditional = ("conditional" in model_name) and ("econditional" not in model_name)
                is_econditional = ("econditional" in model_name)
                is_joint = (eval_type == "joint")

                note_budget = args.prefix_len
                expr_budget = max(1, int(args.prefix_len * args.expr_prefix_scale))

                prefixes = []
                for b in range(batch["seq"].shape[0]):
                    sample_seq = batch["seq"][b]
                    type_field = get_type_events(sample_seq = sample_seq)

                    # Collect event indices for notes and expressive from SOS forward
                    event_positions = torch.arange(start = 0, end = type_field.shape[0], device = type_field.device, dtype = torch.long)
                    # SOS locate (optional)
                    # We assume events start immediately; if explicit SOS events exist in type_field, we could offset here

                    note_idx = (type_field == model_note_token) | (type_field == model_grace_note_token)
                    expr_idx = (type_field == model_expressive_feature_token)

                    note_event_indices = event_positions[note_idx]
                    expr_event_indices = event_positions[expr_idx]

                    if ("baseline" in model_name) and is_joint:
                        # Baseline: first prefix_len notes only (drop expressive)
                        sel_notes = note_event_indices[:note_budget]
                        sel_event_indices = sel_notes
                        # Material = selected notes only
                    elif (is_prefix_model or is_anticipation_model) and (not is_conditional) and (not is_econditional):
                        # Non-conditional prefix/anticipation: first prefix_len notes + optional expressive up to nth note
                        sel_notes = note_event_indices[:note_budget]
                        if sel_notes.numel() > 0:
                            if bool(args.prefix_include_expressive_windowed):
                                nth_note = sel_notes[-1]
                                sel_expr = expr_event_indices[expr_event_indices <= nth_note]
                                sel_event_indices = torch.sort(torch.cat([sel_notes, sel_expr], dim = 0)).values
                            else:
                                sel_event_indices = sel_notes
                        else:
                            sel_event_indices = sel_notes
                    elif is_conditional and (not is_econditional):
                        # Conditional (expressive controls): first prefix_len notes + ALL expressive
                        sel_notes = note_event_indices[:note_budget]
                        sel_expr = expr_event_indices
                        sel_event_indices = torch.sort(torch.cat([sel_notes, sel_expr], dim = 0)).values
                    elif is_econditional:
                        # Econditional (note controls): first expr_budget expressive + ALL notes
                        sel_expr = expr_event_indices[:expr_budget]
                        sel_notes = note_event_indices
                        sel_event_indices = torch.sort(torch.cat([sel_expr, sel_notes], dim = 0)).values
                    else:
                        # Fallback: use first prefix_len notes
                        sel_event_indices = note_event_indices[:note_budget]

                    # Map event indices to token positions
                    token_positions = (sel_event_indices * model_model.decoder.net.n_tokens_per_event) + model_type_dim
                    # For safety, clamp to valid token starts
                    token_positions = token_positions.clamp(min = model_type_dim, max = batch["seq"].shape[1] - model_model.decoder.net.n_tokens_per_event)

                    # Reconstruct event token slices chronologically
                    parts = []
                    for pos in token_positions.tolist():
                        start = pos
                        end = pos + model_model.decoder.net.n_tokens_per_event
                        if model_unidimensional:
                            parts.append(sample_seq[start:end])
                        else:
                            parts.append(sample_seq[start:end, :])
                    if len(parts) == 0:
                        # fallback to SOS-only event
                        sos_event = torch.tensor(data = [model_sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long).reshape(1, 1, len(encoding["dimensions"]))
                        if model_unidimensional:
                            sos_np = sos_event.cpu().numpy()
                            for dimension_index in range(sos_np.shape[-1]):
                                sos_np[..., dimension_index] = model_unidimensional_encoding_function(code = sos_np[..., dimension_index], dimension_index = dimension_index)
                            sos_np = sos_np[..., model_unidimensional_encoding_order].reshape(1, -1)
                            parts = [torch.from_numpy(sos_np).to(device)]
                        else:
                            parts = [sos_event.to(device).reshape(model_model.decoder.net.n_tokens_per_event, len(encoding["dimensions"]))]

                    prefix_sample = torch.cat(parts, dim = 0)
                    prefixes.append(prefix_sample)

                # Pad batch
                prefix = pad(data = prefixes).to(device)

                ##################################################

                # GENERATION
                ##################################################

                # Debug: Log generation parameters
                logging.debug(f"Generation parameters:")
                logging.debug(f"  seq_len: {args.seq_len}")
                logging.debug(f"  prefix shape: {prefix.shape}")
                logging.debug(f"  joint: {is_joint}")
                logging.debug(f"  notes_are_controls: {model_notes_are_controls}")
                logging.debug(f"  is_anticipation: {model_is_anticipation}")
                logging.debug(f"  sigma: {model_sigma}")
                
                # generate new samples
                generated = model_model.generate(
                    seq_in = prefix,
                    seq_len = args.seq_len,
                    eos_token = model_eos,
                    temperature = args.temperature,
                    filter_logits_fn = args.filter,
                    filter_thres = args.filter_threshold,
                    monotonicity_dim = ("type", "time" if model_use_absolute_time else "beat"),
                    joint = is_joint,
                    notes_are_controls = model_notes_are_controls,
                    is_anticipation = model_is_anticipation,
                    sigma = model_sigma
                )
                
                # Debug: Log generation results
                logging.debug(f"Generated shape: {generated.shape}")
                logging.debug(f"Expected generation length: {args.seq_len - prefix.shape[1]}")
                logging.debug(f"Actual generation length: {generated.shape[1]}")

                # concatenate generation to prefix
                generated = torch.cat(tensors = (prefix, generated), dim = 1).cpu().numpy() # wrangle a bit

                # Convert tensors to numpy arrays before multiprocessing
                prefix_numpy = prefix.cpu().numpy()
                generated_numpy = generated
                
                # process each sample in the batch
                def process_helper(j: int):
                    # Get the prefix and generated data for this sample
                    prefix_data = unpad_prefix(prefix = prefix_numpy[j], sos_token = model_sos, pad_value = model_model.decoder.pad_value, n_tokens_per_event = model_model.decoder.net.n_tokens_per_event)
                    generated_data = unpad_prefix(prefix = generated_numpy[j], sos_token = model_sos, pad_value = model_model.decoder.pad_value, n_tokens_per_event = model_model.decoder.net.n_tokens_per_event)
                    
                    # Analyze generation results (prefix vs full sequence)
                    analyze_generation_results(
                        prefix_data = prefix_data,
                        generated_data = generated_data,
                        encoding = encoding,
                        model_name = model_name,
                        stem = f"{stem}_{j}"
                    )
                    
                    return process_single_sample(
                        generated_data = generated_data,
                        prefix_data = prefix_data,
                        stem = f"{stem}_{j}",
                        eval_type = eval_type,
                        output_base_dir = model_output_dir,
                        encoding = encoding,
                        sos_token = model_sos,
                        skip_existing = True,
                        unidimensional_decoding_function = model_unidimensional_decoding_function
                    )
                
                # Process samples in parallel
                with multiprocessing.Pool(processes = args.jobs) as pool:
                    results = pool.map(func = process_helper, iterable = range(len(generated)), chunksize = chunk_size)
                
                # Log results and accumulate counts
                success_count = sum(success for _, success in results)
                total_samples_processed += len(generated)
                total_successful_samples += success_count
                
                # Update model-specific counters
                model_results[model_name]["total_processed"] += len(generated)
                model_results[model_name]["total_successful"] += success_count
                
                # Update progress bar
                pbar.update(len(generated))
            
        # close progress bar and log final results for this model
        pbar.close()
        logging.info(f"Model {model_name}: Processed {model_results[model_name]['total_successful']}/{model_results[model_name]['total_processed']} samples successfully")
        logging.info(LINE)

    logging.info("Sample generation completed!")
    
    # Display final summary for all models
    logging.info("\nGenerated Demo Samples Summary:")
    total_all_processed = 0
    total_all_successful = 0
    
    for model_name, results in model_results.items():
        processed = results["total_processed"]
        successful = results["total_successful"]
        eval_type = results["eval_type"]
        success_rate = (successful / processed * 100) if processed > 0 else 0
        
        total_all_processed += processed
        total_all_successful += successful
        
        logging.info(f"  - {model_name} ({eval_type}): {successful}/{processed} ({success_rate:3.1f}%)")
    

##################################################
