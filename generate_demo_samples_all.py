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
from os import makedirs, get_terminal_size
from typing import Callable, Tuple
import multiprocessing

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

LINE = "-" * get_terminal_size().columns

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
        if skip_existing and all(exists(fp) for fp in [wav_filepath, midi_filepath, mxl_filepath]):
            return True
        
        # Write files
        music.write(wav_filepath, kind = "audio")
        music.write(midi_filepath, kind = "midi")
        music.write(mxl_filepath, kind = "musicxml")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to write music files for {prefix}: {e}")
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
        
        # Decode the full generated sequence
        generated_music = decode.decode(
            codes = generated_data,
            encoding = encoding,
            infer_metrical_time = True,
            unidimensional_decoding_function = unidimensional_decoding_function
        )
        
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
            # Decode the prefix sequence
            prefix_music = decode.decode(
                codes = prefix_data,
                encoding = encoding,
                infer_metrical_time = True,
                unidimensional_decoding_function = unidimensional_decoding_function
            )
            
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

    # create output directory structure
    # Determine experiment type from paths or use default
    experiment_dir = dirname(args.paths)
    experiment_type = basename(experiment_dir)  # Default, could be determined from args.paths
    output_base_dir = f"{args.output_dir}/{experiment_type}"
    
    # make sure the output directory exists
    if not exists(output_base_dir):
        makedirs(output_base_dir)
    
    # Define the 7 model configurations
    with open(f"{experiment_dir}/models/models.txt", "r") as f:
        MODEL_CONFIGS = [line.strip() for line in f.readlines() if line.strip()]

    # load the encoding
    encoding_filepath = f"{experiment_dir}/encoding.json"
    encoding = representation.load_encoding(filepath = encoding_filepath) if exists(encoding_filepath) else representation.get_encoding()

    # set up the logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = f"{output_base_dir}/generate_demo_samples_all.log", mode = "a"), logging.StreamHandler(stream = sys.stdout)])

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
    for model_name in MODEL_CONFIGS:
        logging.info(LINE)
        logging.info(f"Processing model: {model_name}")
        
        # determine the eval type for this model
        eval_type = determine_model_config(model_name)
        logging.info(f"  Using eval type: {eval_type}")
        
        # create model output directory
        model_output_dir = f"{output_base_dir}/{model_name}"
        if not exists(model_output_dir):
            makedirs(model_output_dir)
        
        # load model-specific training arguments
        model_train_args_filepath = f"{experiment_dir}/models/{model_name}/train_args.json"
        if not exists(model_train_args_filepath):
            logging.warning(f"Training arguments not found for {model_name}, skipping...")
            continue
            
        logging.info(f"Loading training arguments from: {model_train_args_filepath}")
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
        model_event_tokens = torch.tensor(data = (model_note_token, model_grace_note_token) if (not model_notes_are_controls) else (model_expressive_feature_token,), device = device)
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
            for i in tqdm(iterable = range(n_iterations), desc = f"Generating samples for {model_name}"):
                
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
                        current_event_type = batch["seq"][seq_index, j + model_type_dim] if model_unidimensional else batch["seq"][seq_index, j, model_type_dim]
                        if (n_events_so_far[seq_index] > args.prefix_len) or (current_event_type == model_eos): # make sure the prefix isn't too long, or if end of song token, no end of song tokens in prefix
                            last_prefix_indicies[seq_index] = j - model_model.decoder.net.n_tokens_per_event
                            break
                        elif current_event_type in model_event_tokens: # if an event
                            n_events_so_far[seq_index] += 1 # increment
                            last_prefix_indicies[seq_index] = j # update last prefix index
                        elif current_event_type == model_sos:
                            last_sos_token_indicies[seq_index] = j
                prefix_conditional_default = [batch["seq"][seq_index, last_sos_token_indicies[seq_index]:(last_prefix_indicies[seq_index] + model_model.decoder.net.n_tokens_per_event)] for seq_index in range(len(last_prefix_indicies))] # truncate to last prefix for each sequence in batch
                for seq_index in range(len(prefix_conditional_default)):
                    if len(prefix_conditional_default[seq_index]) == 0: # make sure the prefix conditional default is not just empty
                        prefix_conditional_default[seq_index] = prefix_default[0]
                prefix_conditional_default = pad(data = prefix_conditional_default).to(device) # pad

                # determine prefix based on model type
                joint = (eval_type == "joint")
                if joint: # joint
                    prefix = prefix_default
                else: # conditional
                    conditional_type, generation_type = eval_type.split("_")[1:]
                    if conditional_type == CONDITIONAL_TYPES[1]: # conditional on notes only
                        prefix = pad(data = [prefix_conditional[(model_get_type_field(prefix_conditional = prefix_conditional) != model_expressive_feature_token)] for prefix_conditional in prefix_conditional_default]).to(device)
                    elif conditional_type == CONDITIONAL_TYPES[2]: # conditional on expressive features only
                        prefix = pad(data = [prefix_conditional[torch.logical_and(input = (model_get_type_field(prefix_conditional = prefix_conditional) != model_note_token), other = (model_get_type_field(prefix_conditional = prefix_conditional) != model_grace_note_token))] for prefix_conditional in prefix_conditional_default]).to(device)
                    else: # conditional on everything
                        prefix = prefix_conditional_default

                # skip irrelevant eval types
                if model_conditional_on_controls and (
                    joint or 
                    (((model_notes_are_controls) and (generation_type in (GENERATION_TYPES[0], GENERATION_TYPES[1]))) or # notes are controls and generation is notes
                     ((not model_notes_are_controls) and (generation_type == (GENERATION_TYPES[0], GENERATION_TYPES[2]))) # expressive features are controls and generation is expressive features
                    )):
                    continue

                ##################################################

                # GENERATION
                ##################################################

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
            
            # Log final results for this model
            logging.info(f"Model {model_name}: Processed {total_successful_samples}/{total_samples_processed} samples successfully")

    logging.info("Sample generation completed!")
    logging.info(LINE)

##################################################
