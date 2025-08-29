# README
# Phillip Long
# August 28, 2025

# Create a baseline model for expression tagging that basically for each field, the model just guesses the most common value for that field.

# IMPORTS
##################################################

import argparse
from os import makedirs
from os.path import exists, dirname
from shutil import rmtree
import sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import pickle
from collections import Counter
import multiprocessing
from typing import Dict, Tuple, List

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from train import NA_VALUE, ALL_STRING, DEFAULT_MAX_SEQ_LEN, MASKS
from dataset import MusicDataset
import representation
import encode

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/deepfreeze/pnlong/muspy_express/experiments/metrical"
PATHS_TRAIN = f"{DATA_DIR}/train.txt"
PATHS_VALID = f"{DATA_DIR}/valid.txt"
ENCODING_FILEPATH = f"{DATA_DIR}/encoding.json"
OUTPUT_DIR = f"{DATA_DIR}/econditional_baseline"

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("-pt", "--paths_train", default = PATHS_TRAIN, type = str, help = ".txt file with absolute filepaths to training dataset.")
    parser.add_argument("-pv", "--paths_valid", default = PATHS_VALID, type = str, help = ".txt file with absolute filepaths to validation dataset.")
    parser.add_argument("-e", "--encoding", default = ENCODING_FILEPATH, type = str, help = ".json file with encoding information.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory that contains any necessary files/subdirectories (such as model checkpoints) created at runtime")
    # data
    parser.add_argument("-bs", "--batch_size", default = 256, type = int, help = "Batch size")
    parser.add_argument("-c", "--conditioning", default = encode.DEFAULT_CONDITIONING, choices = encode.CONDITIONINGS, type = str, help = "Conditioning type")
    parser.add_argument("-s", "--sigma", default = encode.SIGMA, type = float, help = "Sigma anticipation value (for anticipation conditioning, ignored when --conditioning != 'anticipation')")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether to reset the output directory")
    parser.add_argument("-j", "--jobs", default = 10, type = int, help = "Number of workers for data loading")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # LOAD UP MODEL
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # check filepath arguments
    if not exists(args.paths_train):
        raise ValueError("Invalid --paths_train argument. File does not exist.")
    if not exists(args.paths_valid):
        raise ValueError("Invalid --paths_valid argument. File does not exist.")
    
    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # modify output directory for the conditioning type
    args.output_dir += f"/{args.conditioning}"
    if args.reset and exists(args.output_dir):
        rmtree(args.output_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # set conditional mask type
    conditional_mask_type = MASKS[2] # conditional mask type for expression tagging

    # get stuff from encoding
    expressive_feature_code = encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]
    type_idx = encoding["dimensions"].index("type")

    ##################################################


    # TRAINING PROCESS
    ##################################################

    # create a file for the model learned from the training data
    model_output_filepath = f"{args.output_dir}/model.pkl"

    # learn the model from the training data
    if not exists(model_output_filepath):

        # get the dataset
        dataset = MusicDataset(paths = args.paths_train, encoding = encoding, conditioning = args.conditioning, controls_are_notes = True, sigma = args.sigma, is_baseline = False, max_seq_len = DEFAULT_MAX_SEQ_LEN, use_augmentation = False, unidimensional = False)
        data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset.collate)

        # helper function to process a batch (convert tensors to numpy for serialization)
        # def process_batch(batch_seq: np.ndarray) -> Counter:
        #     """
        #     Process a batch of data.

        #     Parameters
        #     ----------
        #     batch_seq : np.ndarray
        #         A numpy array of sequences.

        #     Returns
        #     -------
        #     Counter
        #         A counter for the batch.
        #     """
        #     counters = {dim: Counter() for dim in encoding["dimensions"]} # initialize counters for each dimension            
        #     for seq in batch_seq: # iterate through sequences in the batch
        #         mask = seq[:, type_idx] == expressive_feature_code # filter for expression text using numpy operations
        #         seq_filtered = seq[mask]
        #         for i, dim in enumerate(encoding["dimensions"]): # iterate through dimensions
        #             counters[dim].update(seq_filtered[:, i])
        #     return counters # return counters

        # prepare data for multiprocessing (convert tensors to numpy)
        # batches = [batch["seq"].cpu().numpy() for batch in tqdm(iterable = data_loader, desc = "Preparing training data")]
        
        # use multiprocessing to iterate through batches
        # with multiprocessing.Pool(processes = args.jobs) as pool:
        #     processed_batches = tqdm(iterable = pool.imap_unordered(
        #         func = process_batch,
        #         iterable = batches,
        #         chunksize = 1,
        #     ),
        #     desc = "Training",
        #     total = len(batches))

        # initialize empty counters
        counters = {dim: Counter() for dim in encoding["dimensions"]}
        # for processed_batch in processed_batches: # combine batches
        #     for dim, counter in processed_batch.items():
        #         counters[dim].update(counter)

        # iterate through batches
        for batch in tqdm(iterable = data_loader, desc = "Training", total = len(data_loader)):
            for seq in batch["seq"]: # iterate through sequences in the batch
                seq = seq[seq[:, type_idx] == expressive_feature_code] # filter just for expression text
                for i, dim in enumerate(counters.keys()): # iterate through dimensions
                    counters[dim].update(seq[:, i])

        # from the counters, get the most common value for each dimension
        model = [max(counter.keys(), key = lambda x: counter[x]) for counter in counters.values()]
        with open(model_output_filepath, mode = "wb") as file:
            pickle.dump(obj = model, file = file)
        print(f"Saved model to {model_output_filepath}.")

        # free up memory
        del dataset, data_loader, counters, model
        # del batches, process_batch

    # inform that model was already "trained" earler
    else:
        print(f"Model already trained. Loading from {model_output_filepath}.")

    ##################################################

    
    # RUN MODEL ON VALIDATION DATA
    ##################################################

    # get results output filepath
    results_output_filepath = f"{args.output_dir}/results.csv"

    # run the model over the validation data if needed to get results
    if not exists(results_output_filepath):

        # get the dataset
        dataset = MusicDataset(paths = args.paths_valid, encoding = encoding, conditioning = args.conditioning, controls_are_notes = True, sigma = args.sigma, is_baseline = False, max_seq_len = DEFAULT_MAX_SEQ_LEN, use_augmentation = False, unidimensional = False)
        data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset.collate)
        with open(model_output_filepath, mode = "rb") as file:
            model = pickle.load(file = file) # load list
            model = np.array(model, dtype = encode.ENCODING_ARRAY_TYPE) # convert to tensor

        # helper function to process a batch (numpy version for multiprocessing)
        # def process_batch(batch_seq: np.ndarray) -> Tuple[Dict[str, int], int]:
        #     """
        #     Process a batch of data.

        #     Parameters
        #     ----------
        #     batch_seq : np.ndarray
        #         A numpy array of sequences.

        #     Returns
        #     -------
        #     Tuple[Dict[str, int], int]
        #         A tuple containing a dictionary of counters for each dimension and the total count of expression text tokens.
        #     """
        #     correct_counts_by_field = {dim: 0 for dim in [ALL_STRING] + encoding["dimensions"]} # correct counts by field
        #     total_count = 0 # total count of expression text tokens
        #     for seq in batch_seq: # iterate through sequences in the batch
        #         mask = seq[:, type_idx] == expressive_feature_code # filter for expression text using numpy operations
        #         seq_filtered = seq[mask]
        #         if seq_filtered.shape[0] > 0: # only process if there are expression text tokens
        #             for i, dim in enumerate(encoding["dimensions"]): # iterate through dimensions
        #                 correct_counts_by_field[dim] += np.sum(seq_filtered[:, i] == model[i])
        #             correct_counts_by_field[ALL_STRING] += np.sum(np.all(seq_filtered == model, axis = 1)) # check if entire rows match the model
        #             total_count += seq_filtered.shape[0] # increment total count
        #     return correct_counts_by_field, total_count # return correct counts by field and total count
                
        # # prepare data for multiprocessing (convert tensors to numpy)
        # batches = [batch["seq"].cpu().numpy() for batch in tqdm(iterable = data_loader, desc = "Preparing validation data")]
                
        # # use multiprocessing to iterate through batches
        # with multiprocessing.Pool(processes = args.jobs) as pool:
        #     processed_batches = tqdm(iterable = pool.imap_unordered(
        #         func = process_batch,
        #         iterable = batches,
        #         chunksize = 1,
        #     ),
        #     desc = "Validating",
        #     total = len(batches))
        
        # initialize counters for correct and total predictions
        correct_counts_by_field = np.zeros(shape = len(encoding["dimensions"]) + 1, dtype = np.int32)
        total_count = 0 # total count of expression text tokens
        # for processed_batch in processed_batches:
        #     correct_counts_by_field_batch, total_count_batch = processed_batch
        #     for dim, count in correct_counts_by_field_batch.items():
        #         correct_counts_by_field[dim] += count
        #     total_count += total_count_batch
        #     del correct_counts_by_field_batch, total_count_batch # clear up memory

        # iterate through batches
        for batch in tqdm(iterable = data_loader, desc = "Validating", total = len(data_loader)):
            for seq in batch["seq"]: # iterate through sequences in the batch
                seq = seq.cpu().numpy()
                seq = seq[seq[:, type_idx] == expressive_feature_code] # filter just for expression text
                correct = (seq == model)
                correct_counts_by_field[1:] += np.sum(correct, axis = 0).item()
                correct_counts_by_field[0] += np.sum(np.all(seq == model, axis = 1), axis = 0).item()
                total_count += seq.shape[0]

        # compute accuracy for each field
        results = pd.DataFrame(data = {
            "field": [ALL_STRING] + encoding["dimensions"],
            "count_correct": correct_counts_by_field,
            "count_total": [total_count] * len(correct_counts_by_field),
            "accuracy": correct_counts_by_field / total_count,
        })
        results.to_csv(path_or_buf = results_output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False)
        print(f"Saved results to {results_output_filepath}.")

        # free up memory
        del dataset, data_loader, model, correct_counts_by_field, total_count, results
        # del batches, process_batch

    # inform that results were already "calculated" earler
    else:
        print(f"Results already calculated. Loading from {results_output_filepath}.")

    ##################################################

    
    # STATISTICS AND CONCLUSION
    ##################################################

    # print out results
    results = pd.read_csv(filepath_or_buffer = results_output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False)
    print("")
    print("=" * 100)
    print(f"Results for {args.conditioning} conditioning:")
    for field, accuracy in zip(results["field"], results["accuracy"]):
        print(f"  - {field.title()}: {accuracy * 100:.2f}%")
    del results

    ##################################################

##################################################
