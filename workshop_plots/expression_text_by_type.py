# README
# Phillip Long
# August 9, 2025

# Make plots of expression text types in PDMX.

# IMPORTS
##################################################

import pandas as pd
from multiprocessing import cpu_count
import argparse
from os.path import exists
from os import mkdir
from re import sub
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from extract_expression_text import OUTPUT_DIR
from extract_expression_text_types import EXPRESSION_TEXT_TYPE_DATASET_NAME

##################################################


# CONSTANTS
##################################################

# data constants
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

# seaborn theme
SNS_THEME_KWARGS = {
    "style": "whitegrid",
    "palette": None,
    "font": "serif",
    "rc": dict(),
}
sns.set_theme(**SNS_THEME_KWARGS)

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

# function to split camel case into separate words
def split_camel_case(text: str) -> str:
    """
    Split a camel case string into separate words.

    Parameters
    ----------
    text : str
        String to split

    Returns
    -------
    str
        String with camel case split into separate words
    """
    return sub(pattern = r"([a-z])([A-Z])", repl = r"\1 \2", string = text) # find instances where a lowercase letter is followed by an uppercase letter and insert a space after the lowercase letter

##################################################


# PLOT EXPRESSION TEXT TYPES
##################################################

def plot_expression_text_types_boxplot(data: pd.DataFrame, output_filepath: str):
    """
    Plot expression text types as a boxplot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with expression text types, assumed to have two columns: ["expression_text_type", DURATION_COLUMN_TO_USE]
    output_filepath : str
        Path to output file
    """
    
    # create the horizontal boxplot
    plt.figure(figsize = (10, 8))

    # get order by mean duration (ascending)
    sorted_types = data.groupby(by = "expression_text_type").mean().sort_values(by = DURATION_COLUMN_TO_USE, ascending = True).index.tolist()

    # make boxplot
    sns.boxplot(data = data, x = DURATION_COLUMN_TO_USE, y = "expression_text_type", orient = "h", order = sorted_types)
    
    # format the y-axis labels after plotting
    ax = plt.gca()
    y_labels = [split_camel_case(label.get_text()) for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels = y_labels)
    
    # set labels
    plt.xlabel(get_x_label(duration_column_to_use = DURATION_COLUMN_TO_USE))
    plt.ylabel("Expression Text Type")
    
    # adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # save the plot
    plt.savefig(output_filepath, dpi = LARGE_PLOTS_DPI, bbox_inches = "tight")
    plt.close()
    

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

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
        parser.add_argument("--input_filepath", type = str, default = f"{OUTPUT_DIR}/{EXPRESSION_TEXT_TYPE_DATASET_NAME}.csv", help = "Path to input file.")
        parser.add_argument("--include_lyrics", action = "store_true", help = "Include lyrics in the plot.")
        parser.add_argument("--jobs", type = int, default = int(cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.output_dir):
            raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        elif not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # create plots directory
    plots_dir = f"{args.output_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False, usecols = ["expression_text_type", "duration_beats"])
    print("Completed reading in input data.")

    # filter out lyrics if not included
    if not args.include_lyrics:
        dataset = dataset[dataset["expression_text_type"] != "Lyric"]

    # make duration boxplot
    print("Making expression text type durations boxplot...")
    plot_expression_text_types_boxplot(data = dataset, output_filepath = f"{plots_dir}/expression_text_type_durations" + ("_with_lyrics" if args.include_lyrics else "") + ".pdf")
    print("Completed making expression text type durations boxplot.")

##################################################