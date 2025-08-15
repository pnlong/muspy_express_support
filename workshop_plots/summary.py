# README
# Phillip Long
# August 9, 2025

# Summarize expression text in PDMX.

# IMPORTS
##################################################

import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
from os.path import exists
import pickle
from typing import List
from collections import Counter

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from extract_expression_text import OUTPUT_DIR, DATASET_NAME, NA_VALUE

##################################################


# CONSTANTS
##################################################

OUTPUT_COLUMN_NAMES = ["path", "n_tracks", "n_tracks_with_expression_text_fraction", "has_expression_text", "mean_n_expression_text", "total_n_expression_text", "n_tracks_with_lyrics_fraction", "has_lyrics", "mean_n_lyrics", "total_n_lyrics", "mean_expression_text_density", "mean_expression_text_sparsity", "mean_expression_text_duration", "most_common_expression_text_type"]

##################################################


# HELPER FUNCTIONS
##################################################



##################################################


# MAIN SUMMARIZER FUNCTION
##################################################

def summarize(song_output: List[dict]) -> dict:
    """
    Summarize expression text from a single PDMX entry.

    Parameters
    ----------
    song_output : List[dict]
        List of track dictionaries from a single PDMX entry

    Returns
    -------
    dict
        Summary of expression text from a single PDMX entry that can easily be converted to an output row
    """

    # get number of tracks
    n_tracks = len(song_output)

    # split expression text and lyrics
    expression_text = [track_output["expression_text"][track_output["expression_text"]["type"] != "Lyric"] for track_output in song_output]
    lyrics = [track_output["expression_text"][track_output["expression_text"]["type"] == "Lyric"] for track_output in song_output]

    # get number of expression text
    n_expression_text_per_track = [len(track_expression_text) for track_expression_text in expression_text]
    n_expression_text = sum(n_expression_text_per_track)
    n_tracks_with_expression_text = sum([n_expression_text > 0 for track_n_expression_text in n_expression_text_per_track])

    # get number of lyrics
    n_lyrics = sum([len(track_lyrics) for track_lyrics in lyrics])

    # mean expression text density, number of expression text divided by length of track
    mean_expression_text_density = sum([track_n_expression_text / track_output["track_length"]["seconds"] for track_output, track_n_expression_text in zip(song_output, n_expression_text_per_track)]) / n_tracks # number of expression text divided by length of track
    
    # mean expression text sparsity, seconds between each expression text
    mean_expression_text_sparsity = [0] * n_tracks
    for i in range(n_tracks):
        track_expression_text = expression_text[i]
        track_expression_text = track_expression_text.sort_values(by = "time.s")
        mean_expression_text_sparsity[i] = track_expression_text["time.s"].diff(axis = 0, periods = 1).iloc[1:].mean() if len(track_expression_text) > 1 else 0.0
    mean_expression_text_sparsity = sum(mean_expression_text_sparsity) / n_tracks

    # mean expression text duration, mean of mean expression text durations per track
    mean_expression_text_duration = sum([track_expression_text["duration.s"].mean() for track_expression_text in expression_text]) / n_tracks # mean of mean expression text durations per track

    # most common expression text type
    most_common_expression_text_type = Counter(sum([track_expression_text["type"].tolist() for track_expression_text in expression_text], [])).most_common(1)[0][0]

    # return dictionary
    return {
        "n_tracks": n_tracks,
        "n_tracks_with_expression_text_fraction": n_tracks_with_expression_text / n_tracks,
        "has_expression_text": n_expression_text > 0,
        "mean_n_expression_text": n_expression_text / n_tracks,
        "total_n_expression_text": n_expression_text,
        "has_lyrics": n_lyrics > 0,
        "mean_n_lyrics": n_lyrics / n_tracks,
        "total_n_lyrics": n_lyrics,
        "mean_expression_text_density": mean_expression_text_density,
        "mean_expression_text_sparsity": mean_expression_text_sparsity,
        "mean_expression_text_duration": mean_expression_text_duration,
        "most_common_expression_text_type": most_common_expression_text_type,
    }

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse arguments
    def parse_args(args = None, namespace = None):
        """
        Parse command-line arguments for expression text extraction.
        
        Parameters
        ----------
        args : list, optional
            List of argument strings to parse, by default None (uses sys.argv)
        namespace : argparse.Namespace, optional
            Namespace object to populate with parsed arguments, by default None
            
        Returns
        -------
        argparse.Namespace
            Parsed arguments containing paths and options for expression text extraction
            
        Raises
        ------
        FileNotFoundError
            If the specified PDMX file does not exist
        """
        parser = argparse.ArgumentParser(prog = "Summarize Expression Text Statistics", description = "Summarize expression text statistics.") # create argument parser
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Path to output directory.")
        parser.add_argument("--input_filepath", type = str, default = f"{OUTPUT_DIR}/{DATASET_NAME}.csv", help = "Path to input file.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.output_dir):
            raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        elif not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)
    print("Completed reading in input data.")

    # write column names
    output_filepath = f"{args.output_dir}/summary.csv"
    print("Writing column names...")
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        already_completed_paths = set()
        print(f"Wrote column names to {output_filepath}, no paths have been completed yet.")
    else:
        already_completed_paths = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["path"])["path"])
        print(f"Column names already written to {output_filepath}, {len(already_completed_paths)} paths have been completed.")

    # determine paths to complete
    print("Determining paths to complete...")
    indices_to_complete = [i for i, path_expression in enumerate(dataset["path_expression"]) if path_expression not in already_completed_paths]
    del already_completed_paths
    print(f"{len(indices_to_complete)} paths remaining to complete.")

    ##################################################


    # MAIN FUNCTION
    ##################################################

    def summarize_helper(i: int):
        """
        Summarize expression text from a single PDMX entry.

        Parameters
        ----------
        song_output : List[dict]
            List of track dictionaries from a single PDMX entry

        Returns
        -------
        None
        """

        # load in path expression from pickle file
        path_expression = dataset.at[i, "path_expression"]
        with open(path_expression, mode = "rb") as pickle_file:
            song_output = pickle.load(file = pickle_file)

        # summarize
        summary = summarize(song_output = song_output)
        summary["path"] = path_expression

        # write to csv
        pd.DataFrame(data = [summary], columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")

        # return nothing
        return

    ##################################################


    # EXTRACT DATA WITH MULTIPROCESSING
    ##################################################

    # summarize if necessary
    if len(indices_to_complete) > 0:

        # use multiprocessing
        print("Summarizing expression text...")
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = list(tqdm(iterable = pool.imap_unordered(
                func = summarize_helper,
                iterable = indices_to_complete,
                chunksize = 1
            ),
            desc = "Summarizing",
            total = len(indices_to_complete)))
        print("Summarized expression text.")

    # if already summarized
    else:
        print("Expression text already summarized.")

    # free up memory
    del dataset, indices_to_complete, summarize_helper

    ##################################################


    # LOAD IN DATA, OUTPUT SUMMARY STATISTICS
    ##################################################

    # read in data
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)

    # output summary statistics
    line = "=" * 60
    print(line)
    print("SUMMARY STATISTICS:")

    # expression text
    print(line)
    print(f"Number of songs with expression text: {results['has_expression_text'].sum()}")
    print(f"Percentage of songs with expression text: {results['has_expression_text'].mean() * 100:.2f}%")
    print(f"Average number of expression text per song: {results['total_n_expression_text'].mean():.2f}")
    print(f"Average number of expression text per track: {results['mean_n_expression_text'].mean():.2f}")
    
    # lyrics
    print(line)
    print(f"Number of songs with lyrics: {results['has_lyrics'].sum()}")
    print(f"Percentage of songs with lyrics: {results['has_lyrics'].mean() * 100:.2f}%")
    print(f"Average number of lyrics per song: {results['total_n_lyrics'].mean():.2f}")
    print(f"Average number of lyrics per track: {results['mean_n_lyrics'].mean():.2f}")

    # statistics
    print(line)
    print(f"Mean expression text density: {results['mean_expression_text_density'].mean():.2f}")
    print(f"Mean expression text sparsity: {results['mean_expression_text_sparsity'].mean():.2f}")
    print(f"Mean expression text duration: {results['mean_expression_text_duration'].mean():.2f}")
    print(f"Most common expression text type: {results['most_common_expression_text_type'].mode()[0]}")

    # close
    print(line)

    ##################################################

##################################################