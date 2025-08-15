# README
# Phillip Long
# August 9, 2025

# Make plots of expression text by composer in PDMX.

# IMPORTS
##################################################

import pandas as pd
import numpy as np
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

from extract_expression_text import OUTPUT_DIR, DATASET_NAME, get_id_from_path
from extract_expression_text_types import EXPRESSION_TEXT_TYPE_DATASET_NAME
from expression_text_by_type import SNS_THEME_KWARGS, LARGE_PLOTS_DPI

##################################################


# CONSTANTS
##################################################

# composers
FULL_COMPOSERS = {composer: composer for composer in [
    "Bach", "Beethoven", "Mozart", "Schubert", "Chopin", "Liszt", "Tchaikovsky", "Vivaldi", "Haydn"
]} # allows us to set full, more precise composer names as opposed to just last names
COMPOSERS = list(FULL_COMPOSERS.keys())


# seaborn theme
sns.set_theme(**SNS_THEME_KWARGS)

##################################################


# HELPER FUNCTIONS
##################################################



##################################################


# MATCH TEXT TO COMPOSER
##################################################

# match text to composer
def get_composer_from_text(text: str) -> str:
    """
    Get composer from text.

    Parameters
    ----------
    text : str
        Text to match to composer

    Returns
    -------
    str
        Composer name, None if no match
    """

    # wrangle text
    text = text.lower()
    text = sub(pattern = " ", repl = "", string = text) # remove spaces

    # composer ladder
    if "arcadelt" in text: # is it Jacques Arcadelt?
        return "Arcadelt"
    elif "bach" in text: # if bach in name, probably J.S. Bach, one of his sons, or Jacques Offenbach
        if "offenbach" in text: # if offenbach in name, probably Jacques Offenbach
            return "Offenbach"
        elif ("bach" in text) and (any(keyword in text for keyword in ("johann sebastian", "johan sebastian", "j.s.", "js")) or text == "bach"):
            return "Bach"
        else: # probably one of his sons, who we are not interested in
            return None
    elif "bartok" in text or "bartók" in text: # is it bartok?
        return "Bartók"
    elif "beethoven" in text: # is it beethoven?
        return "Beethoven"
    elif "berlioz" in text: # is it Hector Berlioz?
        return "Berlioz"
    elif "bizet" in text: # is it bizet?
        return "Bizet"
    elif "brahms" in text: # is it brahms?
        return "Brahms"
    elif "bruckner" in text: # is it Anton Bruckner?
        return "Bruckner"
    elif "c418" in text: # is it c418?
        return "C418"
    elif "chopin" in text: # is it chopin?
        return "Chopin"
    elif "czerny" in text: # is it Carl Czerny?
        return "Czerny"
    elif "debussy" in text: # is it debussy?
        return "Debussy"
    elif "dvorak" in text: # is it dvorak?
        return "Dvořák"
    elif "elgar" in text: # is it Edward Elgar?
        return "Elgar"
    elif "foster" in text: # is it Stephen Foster?
        return "Foster"
    elif "gershwin" in text: # is it George Gershwin?
        return "Gershwin"
    elif "grieg" in text: # is it Edvard Grieg?
        return "Grieg"
    elif "handel" in text or "händel" in text: # is it handel?
        return "Handel"
    elif "haydn" in text: # is it haydn?
        return "Haydn"
    elif "hisaishi" in text: # is it Joe Hisaishi?
        return "Hisaishi"
    elif "holst" in text: # is it holst?
        return "Holst"
    elif "joplin" in text: # is it joplin?
        return "Joplin"
    elif "francis" in text and "scott" in text and "key" in text: # is it francis scott key?
        return "Key"
    elif "liszt" in text: # is it liszt?
        return "Liszt"
    elif "lully" in text: # is it Jean-Baptiste Lully?
        return "Lully"
    elif "mahler" in text: # is it Gustav Mahler?
        return "Mahler"
    elif "manchicourt" in text: # is it Pierre de Manchicourt?
        return "Manchicourt"
    elif "mendelssohn" in text: # is it mendelssohn?
        return "Mendelssohn"
    elif "monteverdi" in text: # is it Claudio Monteverdi?
        return "Monteverdi"
    elif "morley" in text: # is it Thomas Morley?
        return "Morley"
    elif "mozart" in text: # is it mozart?
        return "Mozart"
    elif "mussorgsk" in text: # is it Modest Mussorgsky?
        return "Mussorgsky"
    elif "pache" in text: # is it Johann Pachelbel?
        return "Pachelbel"
    elif "paganini" in text: # is it pagani?
        return "Paganini"
    elif "prokofiev" in text: # is it Sergei Prokofiev?
        return "Prokofiev"
    elif "puccini" in text: # is it Giacomo Puccini?
        return "Puccini"
    elif "purcell" in text: # is it Henry Purcell?
        return "Purcell"
    elif "rachmaninoff" in text: # is it Rachmaninoff?
        return "Rachmaninoff"
    elif "ravel" in text: # is it ravel?
        return "Ravel"
    elif "rossini" in text: # is it Gioachino Rossini?
        return "Rossini"
    elif "satie" in text: # is it satie?
        return "Satie"
    elif "saint-s" in text: # is it saint-saëns?
        return "Saint-Saëns"
    elif "scarlatti" in text: # is it scarlatti?
        return "Scarlatti"
    elif "schein" in text: # is it Johann Hermann Schein?
        return "Schein"
    elif "schoenberg" in text: # is it Arnold Schoenberg?
        return "Schoenberg"
    elif "schubert" in text: # is it schubert?
        return "Schubert"
    elif "schumann" in text: # is it schumann?
        return "Schumann"
    elif "scriabin" in text: # is it Alexander Scriabin?
        return "Scriabin"
    elif "shostakovich" in text: # is it Dmitri Shostakovich?
        return "Shostakovich"
    elif "sibelius" in text: # is it Jean Sibelius?
        return "Sibelius"
    elif "skinner" in text: # is it James Scott Skinner?
        return "Skinner"
    elif "strauss" in text: # is it Richard Strauss?
        return "Strauss"
    elif "susato" in text: # is it Tielman Susato?
        return "Susato"
    elif "telemann" in text: # is it Georg Philipp Telemann?
        return "Telemann"
    elif "tárrega" in text: # is it Francisco Tárrega?
        return "Tárrega"
    elif any(keyword in text for keyword in ("tchaikovsky", "tchaikowsky", "tchaikovski", "tchaichovsky")): # is it tchaikovsky?
        return "Tchaikovsky"
    elif "vivaldi" in text: # is it vivaldi?
        return "Vivaldi"
    elif "verdi" in text: # is it Giuseppe Verdi?
        return "Verdi"
    elif "wagner" in text: # is it wagner?
        return "Wagner"

    # no matches, return none
    else:
        return None

##################################################


# PLOT EXPRESSION TEXT BY COMPOSER
##################################################

def plot_expression_text_by_composer_boxplot(data: pd.DataFrame, output_filepath: str):
    """
    Plot expression text by composer.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with expression text types, assumed to have three columns: ["id", "composer", "expression_text_type"]
    output_filepath : str
        Path to output file
    """
    
    # create the horizontal boxplot
    plt.figure(figsize = (10, 8))

    # wrangle data to get expression text counts per song
    data = data.groupby(by = ["id", "composer"]).size().reset_index(name = "count")

    # make boxplot
    sns.boxplot(data = data, x = "id", y = "composer", orient = "h", order = COMPOSERS)
    
    # format the y-axis labels after plotting
    ax = plt.gca()
    y_labels = [FULL_COMPOSERS[label.get_text()] for label in ax.get_yticklabels()] # get full composer name
    ax.set_yticklabels(labels = y_labels)
    
    # set labels
    plt.xlabel("Expression Text Count")
    plt.ylabel("Composer")
    
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
        parser.add_argument("--input_filepath", type = str, default = f"{OUTPUT_DIR}/{DATASET_NAME}.csv", help = "Path to input file.")
        parser.add_argument("--expression_text_types_filepath", type = str, default = f"{OUTPUT_DIR}/{EXPRESSION_TEXT_TYPE_DATASET_NAME}.csv", help = "Path to expression text types file.")
        parser.add_argument("--jobs", type = int, default = int(cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.output_dir):
            raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
        elif not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        elif not exists(args.expression_text_types_filepath):
            raise FileNotFoundError(f"Input file not found: {args.expression_text_types_filepath}")

        return args # return parsed arguments
    args = parse_args()

    # create plots directory
    plots_dir = f"{args.output_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    # determine what composers exist in dataset
    print("Determining composers in dataset...")
    composers_dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False, usecols = ["path_expression", "composer_name"])
    composers_dataset["id"] = list(map(get_id_from_path, composers_dataset["path_expression"])) # get id from path_expression
    composers_dataset["composer"] = list(map(get_composer_from_text, composers_dataset["composer_name"])) # get composer from text
    composers_dataset = composers_dataset[["id", "composer"]] # filter down to just relevant columns
    composers_dataset = composers_dataset[~pd.isna(composers_dataset["composer"])] # filter out rows where composer is None
    print(f"Song count by composer ({len(composers_dataset)} songs total):")
    composer_counts = composers_dataset.groupby(by = "composer").size().reset_index(name = "count")
    composer_counts = composer_counts.sort_values(by = "count", ascending = False)
    for composer, count in zip(composer_counts["composer"], composer_counts["count"]):
        print(f"  - {composer}: {count}")
    print(f"Total number of songs: {composer_counts['count'].sum()}")
    del composer_counts # free up memory

    # filter out rows where composer is not in COMPOSERS
    composers_dataset = composers_dataset[composers_dataset["composer"].isin(COMPOSERS)] # filter out rows where composer is not in COMPOSERS
    composers_dataset = composers_dataset.reset_index(drop = True) # reset index
    print("Completed determining composers in dataset.")

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.expression_text_types_filepath, sep = ",", header = 0, index_col = False, usecols = ["expression_text_type", "duration_beats"])
    print("Completed reading in input data.")

    # wrangle input data
    print("Wrangling input data...")
    dataset = composers_dataset.merge(right = dataset, how = "left", on = "id") # only include ids that are in composers_dataset
    print("Completed wrangling input data.")

    # make duration boxplot
    print("Making expression text by composer boxplot...")
    plot_expression_text_by_composer_boxplot(data = dataset[["id", "composer", "expression_text_type"]], output_filepath = f"{plots_dir}/expression_text_by_composer.pdf")
    print("Completed making expression text by composer boxplot.")

    # output statistics on song count by composer
    line = "=" * 60
    print(line)
    print(f"Song count by composer ({len(composers_dataset)} songs total):")
    composer_counts = composers_dataset.groupby(by = "composer").size().reset_index(name = "count")
    composer_counts = composer_counts.sort_values(by = "count", ascending = False)
    for composer, count in zip(composer_counts["composer"], composer_counts["count"]):
        print(f"  - {composer}: {count}")
    print(f"Total number of songs: {composer_counts['count'].sum()}")

    # output statistics on favorite expression text type by composer
    print(line)
    print("Favorite expression text type by composer:")
    favorite_expression_text_types = dataset.groupby(by = "composer")["expression_text_type"].agg(pd.Series.mode).reset_index(name = "mode")
    for composer, mode in zip(favorite_expression_text_types["composer"], favorite_expression_text_types["mode"]):
        if isinstance(mode, np.ndarray):
            print(f"  - {composer}: {', '.join(mode)}")
        else:
            print(f"  - {composer}: {mode}")

    # close
    print(line)

##################################################