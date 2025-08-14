# README
# Phillip Long
# August 9, 2025

# Make plots of expression text types in PDMX.

# IMPORTS
##################################################

import pandas as pd
import multiprocessing
import tqdm
import argparse
from os.path import exists
import pickle
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from extract_expression_text import OUTPUT_DIR, DATASET_NAME, NA_VALUE

##################################################


# CONSTANTS
##################################################

# data constants
OUTPUT_COLUMN_NAMES = ["song_idx", "track_idx", "expression_text_type", "expression_text_value", "duration_time_steps", "duration_seconds", "duration_beats"]
DURATION_COLUMN_TO_USE = "duration_beats"

# figure constants
LARGE_PLOTS_DPI = 200
ALL_TYPE_NAME = "AllTypes" # name of all features plot name

# colors
TEXT_GREEN = "#5e813f"
TEMPORAL_PURPLE = "#68349a"
SPANNER_BLUE = "#4f71be"
DYNAMIC_GOLD = "#b89230"
SYMBOL_ORANGE = "#ff624c"
SYSTEM_SALMON = "#a91b0d"
ALL_BROWN = "#964b00"
EXPRESSION_TEXT_COLORS = {
    ALL_TYPE_NAME: ALL_BROWN,
    "Lyric": TEXT_GREEN,
    "Tempo": TEMPORAL_PURPLE,
    "Dynamic": DYNAMIC_GOLD,
    "TimeSignature": SYSTEM_SALMON,
    "Text": TEXT_GREEN,
    "KeySignature": SYSTEM_SALMON,
    "Barline": SYSTEM_SALMON,
    "Articulation": SYMBOL_ORANGE,
    "HairPin": DYNAMIC_GOLD,
    "RehearsalMark": TEXT_GREEN,
    "Slur": SPANNER_BLUE,
    "Fermata": TEMPORAL_PURPLE,
    "Pedal": SPANNER_BLUE,
}

##################################################


# HELPER FUNCTIONS
##################################################

# get the x label from the duration column in use
def get_x_label(duration_column_to_use: str) -> str:
    """
    Get the x label from the duration column in use.

    Parameters
    ----------
    duration_column_to_use : str
        The duration column to use

    Returns
    -------
    str
        The x label from the duration column in use
    """
    if duration_column_to_use == "duration_time_steps":
        return "Time Steps"
    elif duration_column_to_use == "duration_seconds":
        return "Seconds"
    elif duration_column_to_use == "duration_beats":
        return "Beats"
    else:
        raise ValueError(f"Invalid duration column to use: {duration_column_to_use}")

##################################################


# MAIN EXTRACT EXPRESSION TEXT TYPES FUNCTION
##################################################

def extract_expression_text_types(song_output: List[dict], song_idx: int) -> dict:
    """
    Extract expression text types from a single PDMX entry.

    Parameters
    ----------
    song_output : List[dict]
        List of track dictionaries from a single PDMX entry
    song_idx : int
        Index of the song in the dataset

    Returns
    -------
    dict
        Dictionary with data for a data frame to be appended to the output csv
    """
    
    # initialize output dictionary
    output = {column: [] for column in OUTPUT_COLUMN_NAMES}
    resolution = song_output[0]["resolution"]

    # go through each track
    for track_idx, track_output in enumerate(song_output):

        # get expression text
        expression_text = track_output["expression_text"]

        # add static columns
        output["song_idx"].extend([song_idx] * len(expression_text))
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


# PLOT EXPRESSION TEXT TYPES
##################################################

def plot_expression_text_types_boxplot(data: pd.DataFrame, output_filepath: str):
    """
    Plot expression text types as a boxplot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with expression text types
    output_filepath : str
        Path to output file
    """
    
    #
    

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
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.output_dir):
            raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        args.input_filepath = f"{args.output_dir}/{DATASET_NAME}.csv"
        if not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)
    print("Completed reading in input data.")

    # write column names
    output_filepath = f"{args.output_dir}/expression_text_type_durations.csv"
    print("Writing column names...")
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        already_completed_indices = set()
        print(f"Wrote column names to {output_filepath}, no paths have been completed yet.")
    else:
        already_completed_indices = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["song_idx"])["song_idx"])
        print(f"Column names already written to {output_filepath}, {len(already_completed_indices)} paths have been completed.")

    # determine paths to complete
    print("Determining paths to complete...")
    indices_to_complete = [i for i in dataset.index if i not in already_completed_indices]
    del already_completed_indices
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
        result = extract_expression_text_types(song_output = song_output, song_idx = i)
        result["path"] = path_expression

        # write to csv
        pd.DataFrame(data = result, columns = OUTPUT_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")

        # return nothing
        return

    ##################################################


    # EXTRACT DATA WITH MULTIPROCESSING
    ##################################################

    # use multiprocessing
    print("Extracting expression text types...")
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
            func = extract_expression_text_types_helper,
            iterable = indices_to_complete,
            chunksize = 1
        ),
        desc = "Extracting Expression Text Types",
        total = len(indices_to_complete)))
    print("Extracted expression text types.")

    ##################################################

##################################################