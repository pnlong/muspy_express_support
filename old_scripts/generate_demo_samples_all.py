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

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

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
        except Exception as e:
            logging.error(f"Failed to decode generated sequence for {stem}: {e}")
            logging.error(f"  Generated data shape: {generated_data.shape}")
            logging.error(f"  Generated data sample: {generated_data[:20] if len(generated_data) > 20 else generated_data}")
            raise
        
        # Write "after" files (full sequence)
        success_after = write_music_files(
            music = generated_music,
            output_dir = output_dir,
            prefix = "after",
            skip_existing = skip_existing
        )
        
        if not success_after:
            return output_dir, False
        
        # Write "before" files only if this is not a joint model (joint models use prefix_default which is just SOS token)
        is_empty = eval_type == "joint"
        if not is_empty:
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
            except Exception as e:
                logging.error(f"Failed to decode prefix sequence for {stem}: {e}")
                logging.error(f"  Prefix data shape: {prefix_data.shape}")
                logging.error(f"  Prefix data sample: {prefix_data[:20] if len(prefix_data) > 20 else prefix_data}")
                raise
            
            # Write "before" files (prefix only)
            success_before = write_music_files(
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

    # set up the logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level = log_level,
        format = "%(message)s",
        handlers = [logging.StreamHandler(stream = sys.stdout)])
    
    if args.debug:
        logging.info("Debug logging enabled - this will provide detailed information about processing steps")

    # create output directory structure
    # Determine experiment type from paths or use default
    experiment_dir = dirname(args.paths)
    experiment_type = basename(experiment_dir)  # Default, could be determined from args.paths
    output_base_dir = f"{args.output_dir}/{experiment_type}"
    if args.reset:
        if exists(output_base_dir):
            logging.info(f"Removing existing output directory because --reset was specified: {output_base_dir}")
            rmtree(output_base_dir)
    
    # make sure the output directory exists
    if not exists(output_base_dir):
        makedirs(output_base_dir)
    
    # Define the 7 model configurations
    with open(f"{experiment_dir}/models/models.txt", "r") as f:
        MODEL_CONFIGS = [line.strip() for line in f.readlines() if line.strip()]
        MODEL_CONFIGS = sorted(MODEL_CONFIGS)

    # load the encoding
    encoding_filepath = f"{experiment_dir}/encoding.json"
    encoding = representation.load_encoding(filepath = encoding_filepath) if exists(encoding_filepath) else representation.get_encoding()

    # log command called and arguments, save arguments
    logging.info(f"Running command: python {' '.join(sys.argv)}")
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
    args_output_filepath = f"{output_base_dir}/generate_demo_samples_all_args.json"
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

                # DETERMINE PREFIX SEQUENCE
                ##################################################

                # default prefix sequence
                prefix_default = torch.repeat_interleave(input = torch.tensor(data = [model_sos] + ([0] * (len(encoding["dimensions"]) - 1)), dtype = torch.long).reshape(1, 1, len(encoding["dimensions"])), repeats = batch["seq"].shape[0], dim = 0).cpu().numpy()
                if model_unidimensional:
                    for dimension_index in range(prefix_default.shape[-1]):
                        prefix_default[..., dimension_index] = model_unidimensional_encoding_function(code = prefix_default[..., dimension_index], dimension_index = dimension_index)
                    prefix_default = prefix_default[..., model_unidimensional_encoding_order].reshape(prefix_default.shape[0], -1)
                prefix_default = torch.from_numpy(prefix_default).to(device)
                n_events_so_far = utils.rep(x = 0, times = len(batch["seq"]))
                last_sos_token_indicies, last_prefix_indicies = utils.rep(x = -1, times = len(batch["seq"])), utils.rep(x = -1, times = len(batch["seq"]))
                for seq_index in range(len(last_prefix_indicies)):
                    for j in range(model_type_dim, batch["seq"].shape[1], model_model.decoder.net.n_tokens_per_event):
                        current_event_type = batch["seq"][seq_index, j] if model_unidimensional else batch["seq"][seq_index, j, model_type_dim]
                        if (n_events_so_far[seq_index] > args.prefix_len) or (current_event_type == model_eos): # make sure the prefix isn't too long, or if end of song token, no end of song tokens in prefix
                            last_prefix_indicies[seq_index] = j - model_model.decoder.net.n_tokens_per_event
                            break
                        elif current_event_type in model_event_tokens: # if an event
                            n_events_so_far[seq_index] += 1 # increment
                            last_prefix_indicies[seq_index] = j # update last prefix index
                        elif current_event_type == model_sos:
                            last_sos_token_indicies[seq_index] = j
                # Debug: Log prefix calculation details
                logging.debug(f"Prefix calculation debug:")
                logging.debug(f"  last_sos_token_indicies: {last_sos_token_indicies}")
                logging.debug(f"  last_prefix_indicies: {last_prefix_indicies}")
                logging.debug(f"  model_event_tokens: {model_event_tokens}")
                logging.debug(f"  model_note_token: {model_note_token}, model_grace_note_token: {model_grace_note_token}")
                logging.debug(f"  model_expressive_feature_token: {model_expressive_feature_token}")
                logging.debug(f"  model_notes_are_controls: {model_notes_are_controls}")
                logging.debug(f"  n_events_so_far: {n_events_so_far}")
                
                # Use simple prefix calculation matching evaluate.py
                prefix_conditional_default = [batch["seq"][seq_index, last_sos_token_indicies[seq_index]:(last_prefix_indicies[seq_index] + model_model.decoder.net.n_tokens_per_event)] for seq_index in range(len(last_prefix_indicies))] # truncate to last prefix for each sequence in batch
                for seq_index in range(len(prefix_conditional_default)):
                    if len(prefix_conditional_default[seq_index]) == 0: # make sure the prefix conditional default is not just empty
                        prefix_conditional_default[seq_index] = prefix_default[0]
                prefix_conditional_default = pad(data = prefix_conditional_default).to(device) # pad

                # determine prefix based on model type
                joint = (eval_type == "joint")
                if joint: # joint
                    prefix = prefix_conditional_default # instead of prefix_default # For prefix and anticipation models, use the extracted prefix instead of just SOS token
                    # prefix = prefix_default # SOS token only, instead of meaningful musical prefix
                else: # conditional
                    conditional_type, generation_type = eval_type.split("_")[1:]
                    
                    # Debug: Log prefix filtering information
                    logging.debug(f"Conditional type: {conditional_type}, Generation type: {generation_type}")
                    logging.debug(f"Model note token: {model_note_token}, grace note token: {model_grace_note_token}, expressive feature token: {model_expressive_feature_token}")
                    
                    if conditional_type == CONDITIONAL_TYPES[1]: # conditional on notes only
                        # Filter out expressive features, keep notes
                        filtered_prefixes = []
                        for i, prefix_conditional in enumerate(prefix_conditional_default):
                            type_field = model_get_type_field(prefix_conditional = prefix_conditional)
                            logging.debug(f"  Prefix {i} type field: {type_field}")
                            mask = (type_field != model_expressive_feature_token)
                            logging.debug(f"  Prefix {i} mask: {mask}")
                            filtered_prefix = prefix_conditional[mask]
                            logging.debug(f"  Prefix {i} filtered shape: {filtered_prefix.shape}")
                            filtered_prefixes.append(filtered_prefix)
                        prefix = pad(data = filtered_prefixes).to(device)
                    elif conditional_type == CONDITIONAL_TYPES[2]: # conditional on expressive features only
                        # Filter out notes, keep expressive features
                        filtered_prefixes = []
                        for i, prefix_conditional in enumerate(prefix_conditional_default):
                            type_field = model_get_type_field(prefix_conditional = prefix_conditional)
                            logging.debug(f"  Prefix {i} type field: {type_field}")
                            mask = torch.logical_and(input = (type_field != model_note_token), other = (type_field != model_grace_note_token))
                            logging.debug(f"  Prefix {i} mask: {mask}")
                            filtered_prefix = prefix_conditional[mask]
                            logging.debug(f"  Prefix {i} filtered shape: {filtered_prefix.shape}")
                            filtered_prefixes.append(filtered_prefix)
                        prefix = pad(data = filtered_prefixes).to(device)
                    else: # conditional on everything
                        prefix = prefix_conditional_default

                ##################################################

                # GENERATION
                ##################################################

                # Debug: Log generation parameters
                logging.debug(f"Generation parameters:")
                logging.debug(f"  seq_len: {args.seq_len}")
                logging.debug(f"  prefix shape: {prefix.shape}")
                logging.debug(f"  joint: {joint}")
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
                    joint = joint,
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
