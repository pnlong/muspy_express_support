# README
# Phillip Long
# November 1, 2023

# Make dataset of musescore files w/ expressive features.

# python /home/pnlong/muspy_express/data_pdmx.py


# IMPORTS
##################################################

import argparse
import pandas as pd
import numpy as np
from os.path import exists, basename, dirname
from os import remove, makedirs
from tqdm import tqdm
import logging
from time import perf_counter, strftime, gmtime
import multiprocessing
import random
from copy import copy
import subprocess
from glob import glob
import warnings
from shutil import rmtree

from os.path import dirname, realpath
import sys
sys.path.insert(0, f"{dirname(realpath(__file__))}")
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}")

import utils
import representation
from encode import extract_data
from muspy2 import muspy as muspy_express

warnings.filterwarnings("ignore", category = muspy_express.inputs.musescore.MuseScoreWarning)

##################################################


# CONSTANTS
##################################################

PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
MUSESCORE_DIR = "/data2/zachary/musescore/data"
OUTPUT_DIR = "/deepfreeze/pnlong/muspy_express/experiments/metrical"
OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n")
TIMEOUT = 20 * 60 # 20 minutes

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Data", description = "Extract Notes and Expressive Features from MuseScore Data.")
    parser.add_argument("--pdmx_filepath", type = str, default = PDMX_FILEPATH, help = "Path to PDMX file.")
    parser.add_argument("--musescore_dir", type = str, default = MUSESCORE_DIR, help = "Path to MuseScore directory.")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-ed", "--explicit_duration", action = "store_true", help = "Whether or not to calculate the 'implied duration' of features without an explicitly-defined duration.")
    parser.add_argument("-v", "--velocity", action = "store_true", help = "Whether or not to include a velocity field that reflects expressive features.")
    parser.add_argument("-a", "--absolute_time", action = "store_true", help = "Whether or not to use absolute (seconds) or metrical (beats) time.")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# EXTRACTION FUNCTION (EXTRACT RELEVANT DATA FROM A GIVEN MUSESCORE FILE
##################################################

def extract(path: str, path_output_prefix: str, use_implied_duration: bool = True, include_velocity: bool = False, use_absolute_time: bool = False) -> tuple:
    """Extract relevant information from a .mscz file, output as tokens

    Parameters
    ----------
    path : str
        Path to the MuseScore file to read.
    path_output_prefix : str
        Prefix to path where tokenized information will be outputted

    Returns
    -------
    tuple: # of tracks processed, # of tokens processed
    """
    
    # LOAD IN MSCZ FILE, CONSTANTS
    ##################################################

    # finish output dictionary
    metadata_path = METADATA.get(path)
    version = VERSION.get(path)

    # try to read musescore
    music = muspy_express.read_musescore(path = path, timeout = TIMEOUT)
    music.realize_annotations()

    # start timer
    start_time = perf_counter()

    ##################################################


    # LOOP THROUGH TRACKS, SCRAPE OBJECTS
    ##################################################
    
    n_tokens = 0
    for i, track in enumerate(music.tracks):

        # do not record if track is drum or is an unknown program
        if track.is_drum or track.program not in representation.KNOWN_PROGRAMS:
            continue
        
        # create MusicExpress object with just one track (we are not doing multitrack)
        track_music = copy(x = music)
        track_music.tracks = [track,]
        data = extract_data(music = track_music, use_implied_duration = use_implied_duration, include_velocity = include_velocity, use_absolute_time = use_absolute_time)

        # create output path from path_output_prefix
        path_output = f"{path_output_prefix}.{i}.npy"

        # save encoded data
        np.save(file = path_output, arr = data)

        # make sure file is valid and can be opened
        try:
            validate = np.load(file = path_output, allow_pickle = True)
            del validate
        except:
            remove(path = path_output)
            continue

        # create current output dictionary; OUTPUT_COLUMNS = ("path", "musescore", "track", "metadata", "version", "n")
        current_output = {
            "path" : path_output,
            "musescore" : path,
            "track" : i,
            "metadata" : metadata_path,
            "version" : version,
            "n" : len(data)
        }

        # update n_tokens
        n_tokens += len(data)

        # write mapping
        pd.DataFrame(
            data = [current_output], columns = OUTPUT_COLUMNS,
        ).to_csv(
            path_or_buf = MAPPING_OUTPUT_FILEPATH,
            mode = "a",
            na_rep = utils.NA_STRING,
            header = False,
            index = False,
        )

        ##################################################

    
    # END STATS
    ##################################################

    end_time = perf_counter()
    total_time = end_time - start_time
    pd.DataFrame(
        data = [{"time": total_time}]
    ).to_csv(
        path_or_buf = TIMING_OUTPUT_FILEPATH,
        mode = "a",
        na_rep = utils.NA_STRING,
        header = False,
        index = False,
    )

    return len(music.tracks), n_tokens

    ##################################################


##################################################


# HELPER FUNCTIONS
##################################################

# get id from path
def get_id_from_path(path: str) -> str:
    """
    Get the id from a path.

    Parameters
    ----------
    path : str
        Path to the original expressive pickle file

    Returns
    -------
    str
        The id from the path
    """
    return basename(path).split(".")[0]

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # CONSTANTS
    ##################################################

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # parse arguments
    args = parse_args()
    output_dir = f"{args.output_dir}/data"
    if exists(output_dir) and args.reset:
        logging.info(f"Resetting output directory: {output_dir}")
        rmtree(output_dir)
        logging.info(f"Output directory reset: {output_dir}")
    if not exists(output_dir): # make output_dir if it doesn't yet exist
        logging.info(f"Creating output directory: {output_dir}")
        makedirs(output_dir)
        logging.info(f"Output directory created: {output_dir}")

    # some constants
    prefix = basename(output_dir)
    TIMING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.timing.txt"
    MAPPING_OUTPUT_FILEPATH = f"{args.output_dir}/{prefix}.csv"

    # load pdmx
    print("Loading PDMX...")
    pdmx = pd.read_csv(filepath_or_buffer = args.pdmx_filepath, sep = ",", header = 0, index_col = False)
    pdmx_dir = dirname(args.pdmx_filepath)
    pdmx = pdmx.drop(columns = ["path", "pdf", "mid"]) # drop everything but mxl
    pdmx = pdmx[pdmx["subset:all_valid"] & pdmx["subset:no_license_conflict"]] # use the correct subset, only valid pdmx paths
    pdmx = pdmx.drop(columns = list(filter(lambda column: column.startswith("subset:"), pdmx.columns))) # drop subset columns
    pdmx = pdmx[pdmx["has_annotations"]] # filter to have annotations
    pdmx = pdmx.reset_index(drop = True) # reset index
    logging.info(f"Loaded PDMX with {len(pdmx)} paths.")       

    # get musescore paths
    logging.info("Determining MuseScore paths...")
    musescore_paths = glob(f"{args.musescore_dir}/**/*.mscz", recursive = True)
    id_to_musescore_path = {id_: path for path, id_ in zip(musescore_paths, map(get_id_from_path, musescore_paths))}
    pdmx["path_mscz"] = pdmx["mxl"].apply(lambda path_mxl: id_to_musescore_path.get(get_id_from_path(path_mxl), None))
    del musescore_paths, id_to_musescore_path
    logging.info("Determined MuseScore paths.")

    # map musescore paths to metadata paths
    paths = pdmx["path_mscz"].tolist() 
    METADATA = {data_path : (metadata_path if not pd.isna(metadata_path) else None) for data_path, metadata_path in zip(pdmx["path_mscz"], pdmx["metadata"])}
    VERSION = {data_path : (version if not pd.isna(version) else None) for data_path, version in zip(pdmx["path_mscz"], pdmx["version"])}

    # write column names to file
    if not exists(MAPPING_OUTPUT_FILEPATH) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = MAPPING_OUTPUT_FILEPATH, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        logging.info(f"Wrote column names to {MAPPING_OUTPUT_FILEPATH}.")
    else:
        logging.info("Loading already-completed paths...")
        completed_paths = set(pd.read_csv(filepath_or_buffer = MAPPING_OUTPUT_FILEPATH, sep = ",", header = 0, index_col = False)["musescore"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining Already-Completed Paths") if path not in completed_paths)
        paths = tuple(random.sample(paths, len(paths)))
        logging.info(f"Loaded {len(completed_paths)} already-completed paths.")

    ##################################################


    # GET PATHS
    ##################################################

    # get prefix for output pickle files
    def get_path_output_prefixes(path: str) -> str:
        # path = "/".join(path.split("/")[-3:]) # get the base name
        # return f"{output_dir}/{path.split('.')[0]}"
        return f"{output_dir}/{basename(path).split('.')[0]}"
    path_output_prefixes = tuple(map(get_path_output_prefixes, paths))
    print("Got paths.")

    ##################################################

    # USE MULTIPROCESSING
    ##################################################

    chunk_size = 1
    start_time = perf_counter() # start the timer
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = pool.starmap(func = extract,
                               iterable = tqdm(iterable = zip(
                                                              paths,
                                                              path_output_prefixes,
                                                              utils.rep(x = not bool(args.explicit_duration), times = len(paths)),
                                                              utils.rep(x = args.velocity, times = len(paths)),
                                                              utils.rep(x = args.absolute_time, times = len(paths)),
                                                              ),
                                               desc = "Extracting Data from MuseScore Files", total = len(paths)),
                               chunksize = chunk_size)
    end_time = perf_counter() # stop the timer
    total_time = end_time - start_time # compute total time elapsed
    total_time = strftime("%H:%M:%S", gmtime(total_time)) # convert into pretty string
    logging.info(f"Total time: {total_time}")
    n_tracks, n_tokens = np.array(results).T.sum(axis = -1)
    del results
    logging.info(f"Total Number of Tracks: {n_tracks:,}")
    logging.info(f"Total Number of Tokens: {n_tokens:,}")

    # make encoding file
    options = (["--velocity",] if args.velocity else []) + (["--absolute_time"] if args.absolute_time else [])
    subprocess.run(args = ["python", f"{dirname(__file__)}/representation.py", "--output_dir", args.output_dir] + options, check = True)

    # split into partitions
    subprocess.run(args = ["python", f"{dirname(__file__)}/split.py", "--input_filepath", MAPPING_OUTPUT_FILEPATH, "--output_dir", args.output_dir], check = True)

    ##################################################

##################################################