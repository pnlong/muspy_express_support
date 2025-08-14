# README
# Phillip Long
# November 1, 2023

# Copied from Herman's MMT: https://github.com/salu133445/mmt/blob/main/mmt/utils.py
# Contains utility (helper) functions


# IMPORTS
##################################################

import json
import pathlib
import warnings
from os.path import exists
from typing import Union, List, Tuple, Collection

import numpy as np

##################################################


# CONSTANTS
##################################################

NA_STRING = "NA"

##################################################


# MISCELLANEOUS FUNCTIONS
##################################################

def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}

# implementation of R's rep function
def rep(x: object, times: int, flatten: bool = False):
    """An implementation of R's rep() function. This can not be used to create a list of empty lists (see https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly)."""
    l = [x] * times
    if flatten:
        l = sum(l, [])
    return l

# unique values in list while retaining order
def unique(l: Union[List, Tuple]) -> list:
    """Returns the unique values from a list while retaining order."""
    return list(dict.fromkeys(list(l)))

# get the product of a list
def product(l: Collection) -> float:
    return np.prod(a = np.array(object = l), axis = 0)
##################################################


# DEAL WITH TEXT
##################################################

# convert camel case to words
def split_camel_case(string: str, sep: str = "-"):
    """Split a camelCase string."""
    splitter = "_"
    if string is not None:
        string = [*string] # convert string to list of characters
        currently_in_digit = False # boolean flag for dealing with numbers
        for i, character in enumerate(string):
            if not character.isdigit() and currently_in_digit: # update whether we are inside of digit
                currently_in_digit = False
            if character.isupper():
                string[i] = splitter + character
            elif character.isdigit() and not currently_in_digit:
                string[i] = splitter + character
                currently_in_digit = True
        words = "".join(string).split(splitter) # convert to list of words
        words = filter(lambda word: word != "", words) # filter out empty words
        return sep.join(words).lower() # join into one string
    return None

##################################################


# SAVING AND LOADING FILES
##################################################

def save_args(filepath: str, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    save_json(filepath = filepath, data = args_dict)


def save_txt(filepath: str, data: list):
    """Save a list to a TXT file."""
    with open(filepath, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filepath: str):
    """Load a TXT file as a list."""
    with open(filepath, encoding = "utf8") as f:
        return [line.strip() for line in f]


def save_json(filepath: str, data: dict):
    """Save data as a JSON file."""
    with open(filepath, "w", encoding = "utf8") as f:
        json.dump(obj = data, fp = f)


def load_json(filepath: str):
    """Load data from a JSON file."""
    with open(filepath, encoding = "utf8") as f:
        return json.load(fp = f)


def save_csv(filepath: str, data, header: str = ""):
    """Save data as a CSV file."""
    np.savetxt(fname = filepath, X = data, fmt = "%d", delimiter = ",", header = header, comments = "")


def load_csv(filepath: str, skiprows: int = 1):
    """Load data from a CSV file."""
    return np.loadtxt(fname = filepath, dtype = int, delimiter = ",", skiprows = skiprows)


# create a csv row
def create_csv_row(info: list, sep: str = ",") -> str:
    """Create a csv row from a list."""
    return sep.join((str(item) if item != None else NA_STRING for item in info)) + "\n"

# write a list to a file
def write_to_file(info: dict, output_filepath: str, columns: list = None):
    """Write a dictionary (representing a row of data) to a file."""
    # if there are provided columns
    if columns is not None:

        # reorder columns if possible
        info = {column: info[column] for column in columns}

        # write columns if they are not there yet
        if not exists(output_filepath):
            with open(output_filepath, "w") as output:
                output.write(create_csv_row(info = columns))

    # write info
    with open(output_filepath, "a") as output:
        output.write(create_csv_row(info = list(info.values())))

##################################################



# DECORATORS
##################################################

def ignore_exceptions(func):
    """Decorator that ignores all errors and warnings."""
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return func(*args, **kwargs)
            except Exception:
                return None
    return inner

##################################################