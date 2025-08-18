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
DISTANCE_COLUMN_TO_USE = "distance_beats"

# figure constants
LARGE_PLOTS_DPI = 200

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

# get the x label from the column in use
def get_x_label(column_to_use: str) -> str:
    """
    Get the x label from the column in use.

    Parameters
    ----------
    column_to_use : str
        The column to use

    Returns
    -------
    str
        The x label from the column in use
    """
    if column_to_use.endswith("time_steps"):
        return "Time Steps"
    elif column_to_use.endswith("seconds"):
        return "Seconds"
    elif column_to_use.endswith("beats"):
        return "Beats"
    else:
        raise ValueError(f"Invalid column to use: {column_to_use}")

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

def plot_expression_text_types_boxplot(data: pd.DataFrame, column_to_use: str, output_filepath: str):
    """
    Plot expression text types as a boxplot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with expression text types, assumed to have two columns: ["expression_text_type", column_to_use]
    column_to_use : str
        Column name to use (either distance or duration for some timing metric)
    output_filepath : str
        Path to output file
    """
    
    # create the horizontal boxplot
    plt.figure(figsize = (6, 4))

    # get order by median duration (descending)
    sorted_types = data.groupby(by = "expression_text_type").median().sort_values(by = column_to_use, ascending = False).index.tolist()

    # make boxplot
    sns.boxplot(data = data, x = column_to_use, y = "expression_text_type", orient = "h", order = sorted_types, showfliers = False)
    
    # format the y-axis labels after plotting
    ax = plt.gca()
    y_labels = [split_camel_case(label.get_text()) for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels = y_labels)
    
    # set labels
    plt.xlabel(get_x_label(column_to_use = column_to_use))
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
        parser.add_argument("--expression_text_types_filepath", type = str, default = f"{OUTPUT_DIR}/{EXPRESSION_TEXT_TYPE_DATASET_NAME}.csv", help = "Path to expression text types file.")
        parser.add_argument("--include_lyrics", action = "store_true", help = "Include lyrics in the plot.")
        parser.add_argument("--jobs", type = int, default = int(cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.output_dir):
            raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        elif not exists(args.expression_text_types_filepath):
            raise FileNotFoundError(f"Input file not found: {args.expression_text_types_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # create plots directory
    plots_dir = f"{args.output_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.expression_text_types_filepath, sep = ",", header = 0, index_col = False, usecols = ["expression_text_type", DURATION_COLUMN_TO_USE, DISTANCE_COLUMN_TO_USE])
    if not args.include_lyrics: # filter out lyrics if not included
        dataset = dataset[dataset["expression_text_type"] != "Lyric"]
    print("Completed reading in input data.")

    # make duration boxplot
    print("Making expression text type durations boxplot...")
    plot_expression_text_types_boxplot(
        data = dataset,
        column_to_use = DURATION_COLUMN_TO_USE,
        output_filepath = f"{plots_dir}/expression_text_by_type_durations" + ("_with_lyrics" if args.include_lyrics else "") + ".pdf",
    )
    print("Completed making expression text type durations boxplot.")

    # make distance boxplot
    print("Making expression text type relative density boxplot...")
    plot_expression_text_types_boxplot(
        data = dataset,
        column_to_use = DISTANCE_COLUMN_TO_USE,
        output_filepath = f"{plots_dir}/expression_text_by_type_relative_density" + ("_with_lyrics" if args.include_lyrics else "") + ".pdf",
    )
    print("Completed making expression text type relative density boxplot.")

##################################################