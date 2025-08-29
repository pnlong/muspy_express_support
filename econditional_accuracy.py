# README
# Phillip Long
# August 28, 2025

# IMPORTS
##################################################

import argparse
import logging
from os.path import exists, isdir
from os import listdir
import pandas as pd

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/deepfreeze/pnlong/muspy_express/experiments/metrical"

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Accuracy", description = "Calculate the accuracy of the expression-conditional-on-notes model's predictions.")
    parser.add_argument("--data_dir", type = str, default = DATA_DIR, help = "Path to data directory.")
    parser.add_argument("--mask_type", type = str, default = "note", choices = ("total", "note", "expressive"), help = "Type of mask to use.")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # parse arguments
    args = parse_args()
    if not exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    # get econditional models
    models_dir = f"{args.data_dir}/models"
    models = [model for model in listdir(models_dir) if isdir(f"{models_dir}/{model}") and "econditional" in model]
    models_dirs = [f"{models_dir}/{model}" for model in models]

    ##################################################


    # GET ACCURACIES
    ##################################################

    for model, model_dir in zip(models, models_dirs):

        # load the model's performance
        performance = pd.read_csv(filepath_or_buffer = f"{model_dir}/performance.csv", sep = ",", header = 0, index_col = False)
        performance = performance[(performance["partition"] == "valid") & (performance["metric"] == "accuracy") & (performance["mask"] == args.mask_type)]

        # get the maximum accuracy for each field
        logging.info(f"Model: {model}")
        for field in pd.unique(performance["field"]):
            max_accuracy = performance[performance["field"] == field]["value"].max()
            logging.info(f"  - {field.title()}: {max_accuracy * 100:.2f}%")

    ##################################################

##################################################