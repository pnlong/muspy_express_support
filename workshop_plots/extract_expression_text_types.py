# README
# Phillip Long
# August 9, 2025

# Make plots of expression text types in PDMX.

# IMPORTS
##################################################

import pandas as pd
import multiprocessing
from tqdm import tqdm
import argparse
from os.path import exists
import pickle
from typing import List
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from extract_expression_text import OUTPUT_DIR, DATASET_NAME, NA_VALUE, get_id_from_path

##################################################


# CONSTANTS
##################################################

# data constants
OUTPUT_COLUMN_NAMES = ["id", "track_idx", "expression_text_type", "expression_text_value", "duration_time_steps", "duration_seconds", "duration_beats"]
EXPRESSION_TEXT_TYPE_DATASET_NAME = "expression_text_type"

##################################################


# HELPER FUNCTIONS
##################################################



##################################################


# MAIN EXTRACT EXPRESSION TEXT TYPES FUNCTION
##################################################

def extract_expression_text_types(song_output: List[dict]) -> dict:
    """
    Extract expression text types from a single PDMX entry.

    Parameters
    ----------
    song_output : List[dict]
        List of track dictionaries from a single PDMX entry

    Returns
    -------
    dict
        Dictionary with data for a data frame to be appended to the output csv
    """
    
    # initialize output dictionary
    output = {column: [] for column in OUTPUT_COLUMN_NAMES}
    if len(song_output) == 0: # ensure song output is non-empty
        raise RuntimeError("Song output is empty.")
    resolution = song_output[0]["resolution"]

    # go through each track
    for track_idx, track_output in enumerate(song_output):

        # get expression text
        expression_text = track_output["expression_text"]

        # add static columns
        output["track_idx"].extend([track_idx] * len(expression_text))
        
        # add dynamic columns
        output["expression_text_type"].extend(expression_text["type"].tolist())
        output["expression_text_value"].extend(expression_text["value"].tolist())
        output["duration_time_steps"].extend(expression_text["duration"].tolist())
        output["duration_seconds"].extend(expression_text["duration.s"].tolist())
        output["duration_beats"].extend((expression_text["duration"] / resolution).tolist())

        # free up memory
        del expression_text

    # return output dictionary
    return output

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
    output_filepath = f"{args.output_dir}/{EXPRESSION_TEXT_TYPE_DATASET_NAME}.csv"
    print("Writing column names...")
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        already_completed_ids = set()
        print(f"Wrote column names to {output_filepath}, no paths have been completed yet.")
    else:
        already_completed_ids = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["id"])["id"])
        print(f"Column names already written to {output_filepath}, {len(already_completed_ids)} paths have been completed.")

    # determine paths to complete
    print("Determining paths to complete...")
    indices_to_complete = [i for i, id_ in enumerate(map(get_id_from_path, dataset["path_expression"])) if id_ not in already_completed_ids]
    del already_completed_ids
    print(f"{len(indices_to_complete)} paths remaining to complete.")

    ##################################################


    # MAIN FUNCTION
    ##################################################

    def extract_expression_text_types_helper(i: int):
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

        # extract expression text types
        result = extract_expression_text_types(song_output = song_output)
        result["id"] = [get_id_from_path(path = path_expression)] * len(result["track_idx"])

        # write to csv
        pd.DataFrame(data = result, columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")

        # return nothing
        return

    ##################################################


    # EXTRACT DATA WITH MULTIPROCESSING
    ##################################################

    # extract expression text types if necessary
    if len(indices_to_complete) > 0:

        # use multiprocessing
        print("Extracting expression text types...")
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = list(tqdm(iterable = pool.imap_unordered(
                func = extract_expression_text_types_helper,
                iterable = indices_to_complete,
                chunksize = 1,
            ),
            desc = "Extracting Expression Text Types",
            total = len(indices_to_complete)))
        print("Extracted expression text types.")

    # expression text types already extracted
    else:
        print("Expression text types already extracted.")

    # free up memory
    del dataset, indices_to_complete, extract_expression_text_types_helper

    ##################################################


    # LOAD IN DATA, SUMMARIZE
    ##################################################

    # read in data
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    results_expression_text = results[results["expression_text_type"] != "Lyric"]
    results_lyrics = results[results["expression_text_type"] == "Lyric"]

    # output summary statistics
    line = "=" * 60
    print(line)
    print("SUMMARY STATISTICS:")

    # counts
    print(line)
    print(f"Number of expression text tokens: {len(results_expression_text)}")
    print(f"Number of lyric tokens: {len(results_lyrics)}")
    expression_text_ids = set(results_expression_text["id"].unique())
    print(f"Number of songs with expression text: {len(expression_text_ids)}")
    lyric_ids = set(results_lyrics["id"].unique())
    print(f"Number of songs with lyrics: {len(lyric_ids)}")
    print(f"Number of songs with both expression text and lyrics: {len(expression_text_ids & lyric_ids)}")
    
    # averages
    print(line)
    results_expression_text_counts = results_expression_text[["id", "track_idx"]].groupby(by = ["id", "track_idx"]).size().reset_index(name = "count")
    print(f"Average number of expression text per song: {results_expression_text_counts[['id', 'count']].groupby(by = 'id').sum().mean():.2f}")
    print(f"Average number of expression text per track: {results_expression_text_counts['count'].mean():.2f}")
    results_lyrics_counts = results_lyrics[["id", "track_idx"]].groupby(by = ["id", "track_idx"]).size().reset_index(name = "count")
    print(f"Average number of lyrics per song: {results_lyrics_counts[['id', 'count']].groupby(by = 'id').sum().mean():.2f}")
    print(f"Average number of lyrics per track: {results_lyrics_counts['count'].mean():.2f}")
    
    # mode
    print(line)
    print("Most Common Expression Text Types:")
    most_common_types = results["expression_text_type"].value_counts(sort = True, ascending = False)
    for expression_text_type in most_common_types.index:
        print(f"  - {expression_text_type}: {most_common_types[expression_text_type]}")

    # close
    print(line)

    ##################################################

##################################################