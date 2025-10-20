#!/usr/bin/env python3
"""
Script to copy demo samples from input directory to output directory based on JSON configuration.
Creates symlinks for .mxl files and copies .wav files.
"""

import argparse
import json
import os
import shutil
import sys
from tqdm import tqdm

# Default configuration
DEFAULT_INPUT_DIR = "/deepfreeze/pnlong/muspy_express/demo_samples"
DEFAULT_CONFIG_FILEPATH = f"{os.path.dirname(os.path.realpath(__file__))}/demo/demo_samples.json"
DEFAULT_OUTPUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/demo/resources"


def get_model_directory_name(model_type, conditioning_strategy, timing_scheme):
    """
    Map JSON model configuration to actual directory names in the input directory.
    
    Args:
        model_type: 'baseline', 'joint', 'conditional', 'econditional'
        conditioning_strategy: 'prefix', 'anticipation', or None for baseline
        timing_scheme: 'metrical' or 'real'
    
    Returns:
        Directory name in the input directory
    """
    if model_type == 'baseline':
        return f"baseline_ape_{'20M' if timing_scheme == 'metrical' else '21M'}"
    
    # For non-baseline models, we need both conditioning strategy and model type
    model_suffix = {
        'joint': 'ape',
        'conditional': 'conditional_ape', 
        'econditional': 'econditional_ape'
    }[model_type]
    
    model_size = '20M' if timing_scheme == 'metrical' else '21M'
    
    if conditioning_strategy == 'prefix':
        return f"prefix_{model_suffix}_{model_size}"
    elif conditioning_strategy == 'anticipation':
        return f"anticipation_{model_suffix}_{model_size}"
    else:
        raise ValueError(f"Unknown conditioning strategy: {conditioning_strategy}")


def create_output_structure(config, output_dir):
    """
    Create the output directory structure based on the JSON config.
    
    Args:
        config: Loaded JSON configuration
        output_dir: Base output directory path
    
    Returns:
        List of tuples (model_type, conditioning_strategy, timing_scheme, generation_id, output_path)
    """
    output_paths = []
    
    for model_type, model_config in config.items():
        if model_type == 'baseline':
            # Baseline has direct metrical/real structure
            for timing_scheme, generation_id in model_config.items():
                output_path = os.path.join(output_dir, model_type, timing_scheme)
                output_paths.append((model_type, None, timing_scheme, generation_id, output_path))
        else:
            # Other models have prefix/anticipation structure
            for conditioning_strategy, timing_config in model_config.items():
                for timing_scheme, generation_id in timing_config.items():
                    output_path = os.path.join(output_dir, model_type, conditioning_strategy, timing_scheme)
                    output_paths.append((model_type, conditioning_strategy, timing_scheme, generation_id, output_path))
    
    return output_paths


def copy_files_and_create_symlinks(input_dir, output_paths, demo_samples_dir):
    """
    Copy .wav files and create symlinks for .mxl files.
    
    Args:
        input_dir: Input directory path
        output_paths: List of output path tuples
        demo_samples_dir: Demo samples directory path
    """
    total_operations = len(output_paths) * 4  # 4 operations per path: copy before.wav, after.wav, symlink before.mxl, after.mxl
    
    with tqdm(total=total_operations, desc="Processing demo samples", unit="files") as pbar:
        for model_type, conditioning_strategy, timing_scheme, generation_id, output_path in output_paths:
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Get source directory
            model_dir_name = get_model_directory_name(model_type, conditioning_strategy, timing_scheme)
            source_dir = os.path.join(input_dir, timing_scheme, model_dir_name, generation_id)
            
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory does not exist: {source_dir}")
                pbar.update(4)  # Skip this iteration
                continue
            
            # Copy .wav files
            for wav_file in ['before.wav', 'after.wav']:
                source_wav = os.path.join(source_dir, wav_file)
                dest_wav = os.path.join(output_path, wav_file)
                
                if os.path.exists(source_wav):
                    shutil.copy2(source_wav, dest_wav)
                    # pbar.set_postfix_str(f"Copied {wav_file}")
                else:
                    print(f"Warning: Source file does not exist: {source_wav}")
                
                pbar.update(1)
            
            # Create symlinks for .mxl files
            for mxl_file in ['before.mxl', 'after.mxl']:
                source_mxl = os.path.join(source_dir, mxl_file)
                dest_mxl = os.path.join(output_path, mxl_file)
                
                if os.path.exists(source_mxl):
                    # Remove existing symlink if it exists
                    if os.path.exists(dest_mxl) and os.path.islink(dest_mxl):
                        os.unlink(dest_mxl)
                    elif os.path.exists(dest_mxl) and not os.path.islink(dest_mxl):
                        os.remove(dest_mxl)
                
                    # copy
                    shutil.copy2(source_mxl, dest_mxl)
                    # pbar.set_postfix_str(f"Copied {mxl_file}")
                else:
                    print(f"Warning: Source file does not exist: {source_mxl}")
                
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Copy demo samples based on JSON configuration')
    
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--config', 
                       type=str, 
                       default=DEFAULT_CONFIG_FILEPATH,
                       help='Path to JSON configuration file')
    
    parser.add_argument('--input_dir', 
                       type=str, 
                       default=DEFAULT_INPUT_DIR,
                       help='Path to input demo samples directory')
    
    parser.add_argument('--output_dir', 
                       type=str, 
                       default=DEFAULT_OUTPUT_DIR,
                       help='Path to output directory')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create demo_samples subdirectory in output_dir
    demo_samples_dir = os.path.join(args.output_dir, 'demo_samples')
    
    print(f"Creating demo_samples directory: {demo_samples_dir}")
    os.makedirs(demo_samples_dir, exist_ok=True)
    
    # Load JSON configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output structure
    print("Creating output directory structure...")
    output_paths = create_output_structure(config, demo_samples_dir)
    
    print(f"Found {len(output_paths)} model configurations to process")
    
    # Copy files and create symlinks
    print("Copying files and creating symlinks...")
    copy_files_and_create_symlinks(args.input_dir, output_paths, demo_samples_dir)
    
    print(f"\nDemo samples successfully processed!")
    print(f"Output directory: {demo_samples_dir}")
    print(f"Total configurations processed: {len(output_paths)}")


if __name__ == '__main__':
    main()
