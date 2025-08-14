# README
# Phillip Long
# August 9, 2025

# Parse expression text from PDMX.

# IMPORTS
##################################################

import pandas as pd
import argparse
import multiprocessing
from os.path import basename, dirname, exists
from os import makedirs, mkdir
from shutil import rmtree
from glob import glob
from tqdm import tqdm
import pickle
from re import sub
from typing import List
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}")

from muspy2 import muspy as muspy_express

##################################################


# CONSTANTS
##################################################

# filepaths
PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
MUSESCORE_DIR = "/data2/zachary/musescore/data"
OUTPUT_DIR = "/deepfreeze/pnlong/muspy_express"

# extra constants
DATASET_NAME = "expression_pickles"
NA_VALUE = "NA"

# issues column names
ISSUES_COLUMNS = ["path", "issue"]

# constants copied from parse_mscz.py for expression extraction
EXPRESSION_TEXT_COLUMNS = ["time", "duration", "time.s", "duration.s", "type", "value"]

# dynamic constants
DYNAMIC_VELOCITY_MAP = {
    "pppppp": 4, "ppppp": 8, "pppp": 12, "ppp": 16, "pp": 33, "p": 49, "mp": 64,
    "mf": 80, "f": 96, "ff": 112, "fff": 126, "ffff": 127, "fffff": 127, "ffffff": 127,
    "sfpp": 96, "sfp": 112, "sf": 112, "sff": 126, "sfz": 112, "sffz": 126, "fz": 112, "rf": 112, "rfz": 112,
    "fp": 96, "pf": 49, "s": 64, "r": 64, "z": 64, "n": 64, "m": 64,
    "dynamic-marking": 64,
}
DYNAMIC_DYNAMICS = set(tuple(DYNAMIC_VELOCITY_MAP.keys())[:tuple(DYNAMIC_VELOCITY_MAP.keys()).index("ffffff")])

# tempo constants
TEMPO_QPM_MAP = {
    "largo": 50, "lento": 60, "adagio": 72, "andante": 92, "moderato": 108,
    "allegretto": 120, "allegro": 156, "vivace": 176, "presto": 200, "prestissimo": float("inf"),
}

# program-to-instrument mapping (copied from representation.py)
PROGRAM_INSTRUMENT_MAP = {
    # Pianos
    0: "piano", 1: "piano", 2: "piano", 3: "piano", 4: "electric-piano", 5: "electric-piano", 6: "harpsichord", 7: "clavinet",
    # Chromatic Percussion  
    8: "celesta", 9: "glockenspiel", 10: "music-box", 11: "vibraphone", 12: "marimba", 13: "xylophone", 14: "tubular-bells", 15: "dulcimer",
    # Organs
    16: "organ", 17: "organ", 18: "organ", 19: "church-organ", 20: "organ", 21: "accordion", 22: "harmonica", 23: "bandoneon",
    # Guitars
    24: "nylon-string-guitar", 25: "steel-string-guitar", 26: "electric-guitar", 27: "electric-guitar", 28: "electric-guitar", 29: "electric-guitar", 30: "electric-guitar", 31: "electric-guitar",
    # Basses
    32: "bass", 33: "electric-bass", 34: "electric-bass", 35: "electric-bass", 36: "slap-bass", 37: "slap-bass", 38: "synth-bass", 39: "synth-bass",
    # Strings
    40: "violin", 41: "viola", 42: "cello", 43: "contrabass", 44: "strings", 45: "strings", 46: "harp", 47: "timpani",
    # Ensemble
    48: "strings", 49: "strings", 50: "synth-strings", 51: "synth-strings", 52: "voices", 53: "voices", 54: "voices", 55: "orchestra-hit",
    # Brass
    56: "trumpet", 57: "trombone", 58: "tuba", 59: "trumpet", 60: "horn", 61: "brasses", 62: "synth-brasses", 63: "synth-brasses",
    # Reed
    64: "soprano-saxophone", 65: "alto-saxophone", 66: "tenor-saxophone", 67: "baritone-saxophone", 68: "oboe", 69: "english-horn", 70: "bassoon", 71: "clarinet",
    # Pipe
    72: "piccolo", 73: "flute", 74: "recorder", 75: "pan-flute", 76: None, 77: None, 78: None, 79: "ocarina",
    # Synth Lead
    80: "lead", 81: "lead", 82: "lead", 83: "lead", 84: "lead", 85: "lead", 86: "lead", 87: "lead",
    # Synth Pad
    88: "pad", 89: "pad", 90: "pad", 91: "pad", 92: "pad", 93: "pad", 94: "pad", 95: "pad",
    # Synth Effects
    96: None, 97: None, 98: None, 99: None, 100: None, 101: None, 102: None, 103: None,
    # Ethnic
    104: "sitar", 105: "banjo", 106: "shamisen", 107: "koto", 108: "kalimba", 109: "bag-pipe", 110: "violin", 111: "shehnai",
    # Percussive
    112: None, 113: None, 114: None, 115: None, 116: None, 117: "melodic-tom", 118: "synth-drums", 119: "synth-drums",
    # Sound effects
    120: None, 121: None, 122: None, 123: None, 124: None, 125: None, 126: None, 127: None,
}
KNOWN_PROGRAMS = [k for k, v in PROGRAM_INSTRUMENT_MAP.items() if v is not None]

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

# tempo mapping functions
def qpm_tempo_mapper(qpm: float) -> str:
    """
    Map quarters per minute (QPM) values to standard tempo markings.
    
    Parameters
    ----------
    qpm : float
        Quarters per minute value, or None for default
        
    Returns
    -------
    str
        Standard tempo marking (e.g., "largo", "allegro", "prestissimo")
    """
    if qpm is None:
        qpm = 120  # default bpm
    for bpm in TEMPO_QPM_MAP.keys():
        if qpm <= TEMPO_QPM_MAP[bpm]:
            return bpm
    return "prestissimo"

# time signature mapping functions
def time_signature_change_mapper(time_signature_change_ratio: float) -> str:
    """
    Map time signature change ratios to standardized string representations.
    
    Parameters
    ----------
    time_signature_change_ratio : float
        Ratio between new and old time signature (new_numerator/new_denominator) / (old_numerator/old_denominator)
        
    Returns
    -------
    str
        Standardized time signature change string (e.g., "time-signature-change-1.0", "time-signature-change-other")
    """
    if abs(time_signature_change_ratio - 1.0) < 1e-3:
        return "time-signature-change-1.0"
    elif abs(time_signature_change_ratio - 0.75) < 1e-3:
        return "time-signature-change-0.75"
    elif abs(time_signature_change_ratio - 1.33) < 1e-3:
        return "time-signature-change-1.33"
    else:
        return "time-signature-change-other"

# text cleanup functions for scraping
def check_text(text: str) -> str:
    """
    Clean up text by removing extra spaces and normalizing punctuation.
    
    Parameters
    ----------
    text : str
        Raw text string to clean up, or None
        
    Returns
    -------
    str or None
        Cleaned text with normalized spacing and punctuation, or None if input was None
    """
    if text is not None:
        return sub(pattern = ": ", repl = ":", string = sub(pattern = ", ", repl = ",", string = " ".join(text.split()))).strip()
    return None

# text cleaning functions
def clean_up_text(text: str) -> str:
    """
    Clean and normalize text for consistent representation in expression text.
    
    Performs camel case splitting, dash replacement, space normalization, 
    alphanumeric extraction, and lowercase conversion.
    
    Parameters
    ----------
    text : str
        Raw text string to clean and normalize, or None
        
    Returns
    -------
    str or None
        Cleaned and normalized text suitable for expression text representation, or None if input was None
    """
    if text is not None:
        text = sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = sub(pattern = "-", repl = " ", string = text)  # get rid of dashes
        text = sub(pattern = " ", repl = "-", string = check_text(text = text))  # replace any whitespace with dashes
        text = sub(pattern = "[^\w-]", repl = "", string = text)  # extract alphanumeric
        return text.lower()  # convert to lower case
    return None

# scraping functions
def scrape_lyrics(lyrics) -> pd.DataFrame:
    """
    Extract lyrics from a list of Lyric objects into a standardized DataFrame.
    
    Parameters
    ----------
    lyrics : list
        List of Lyric objects with time and lyric attributes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing lyric data
    """
    if not lyrics:
        return pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    
    lyrics_encoded = {key: [] for key in EXPRESSION_TEXT_COLUMNS}
    
    for lyric in lyrics:
        if hasattr(lyric, "time") and hasattr(lyric, "lyric"):
            lyrics_encoded["time"].append(lyric.time)
            lyrics_encoded["duration"].append(0)  # lyrics typically don't have duration
            lyrics_encoded["time.s"].append(0)  # placeholder, will be calculated later
            lyrics_encoded["duration.s"].append(0)  # placeholder, will be calculated later
            lyrics_encoded["type"].append("Lyric")
            lyrics_encoded["value"].append(lyric.lyric)
    
    return pd.DataFrame(data = lyrics_encoded, columns = EXPRESSION_TEXT_COLUMNS)

# scrape annotations
def scrape_annotations(annotations, song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """
    Extract annotations (text, dynamics, fermatas, etc.) from music with implied duration calculation.
    
    Processes various annotation types and calculates durations either from explicit duration attributes
    or by computing implied duration as the time until the next annotation of the same type.
    
    Parameters
    ----------
    annotations : list
        List of annotation objects with time, annotation, and optionally duration attributes
    song_length : int
        Total length of the song in time steps, used as end boundary for duration calculation
    use_implied_duration : bool, optional
        Whether to calculate implied durations between annotations of same type, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing annotation data
    """
    desired_annotation_types = ("Text", "TextSpanner", "RehearsalMark", "Dynamic", "HairPinSpanner", "Fermata", "TempoSpanner", "TechAnnotation")
    output_columns = EXPRESSION_TEXT_COLUMNS
    annotations_encoded = {key: [] for key in output_columns}
    
    # group annotations by type for implied duration calculation
    annotations_by_type = {}
    for annotation in annotations:
        annotation_type = annotation.annotation.__class__.__name__
        if annotation_type not in desired_annotation_types:
            continue
        if annotation_type not in annotations_by_type:
            annotations_by_type[annotation_type] = []
        annotations_by_type[annotation_type].append(annotation)
    
    # sort each type by time
    for annotation_type in annotations_by_type:
        annotations_by_type[annotation_type].sort(key = lambda x: x.time)

    # process annotations
    for annotation_type, type_annotations in annotations_by_type.items():
        for i, annotation in enumerate(type_annotations):
            # time
            annotations_encoded["time"].append(annotation.time)
            annotations_encoded["time.s"].append(0)  # placeholder, will be calculated later
            
            # duration calculation
            if hasattr(annotation.annotation, "duration"):
                duration = annotation.annotation.duration
            elif use_implied_duration and annotation_type == "Dynamic" and hasattr(annotation.annotation, "subtype") and annotation.annotation.subtype.lower() not in DYNAMIC_DYNAMICS:
                # hike dynamics get 0 duration even with implied duration
                duration = 0
            elif use_implied_duration:
                # calculate implied duration as time to next annotation of same type or song end
                if i + 1 < len(type_annotations):
                    duration = type_annotations[i + 1].time - annotation.time
                else:
                    duration = song_length - annotation.time
            else:
                duration = 0
            
            annotations_encoded["duration"].append(duration)
            annotations_encoded["duration.s"].append(0)  # placeholder, will be calculated later
            
            # event type
            annotations_encoded["type"].append(annotation.annotation.__class__.__name__)
            
            # deal with value field
            value = None
            if hasattr(annotation.annotation, "text"):
                value = clean_up_text(text = annotation.annotation.text)
            elif hasattr(annotation.annotation, "subtype"):
                value = clean_up_text(text = annotation.annotation.subtype)
            annotations_encoded["value"].append(value or "annotation")
    
    return pd.DataFrame(data = annotations_encoded, columns = output_columns)

# scrape barlines
def scrape_barlines(barlines, song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """
    Extract special barlines (excluding single and repeat barlines) with implied duration calculation.
    
    Filters out regular single barlines and repeat barlines, then processes special barlines
    like double barlines, calculating their durations until the next barline or song end.
    
    Parameters
    ----------
    barlines : list
        List of barline objects with time and subtype attributes
    song_length : int
        Total length of the song in time steps, used as end boundary for duration calculation
    use_implied_duration : bool, optional
        Whether to calculate implied durations between barlines, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing barline data
    """
    output_columns = EXPRESSION_TEXT_COLUMNS
    # filter out single barlines and repeat barlines
    filtered_barlines = [b for b in barlines if not ((hasattr(b, "subtype") and b.subtype == "single") or (hasattr(b, "subtype") and "repeat" in b.subtype.lower()))]
    
    if not filtered_barlines:
        return pd.DataFrame(columns = output_columns)
    
    barlines_encoded = {key: [None] * len(filtered_barlines) for key in output_columns}
    
    for i, barline in enumerate(filtered_barlines):
        barlines_encoded["time"][i] = barline.time
        barlines_encoded["time.s"][i] = 0  # placeholder, will be calculated later
        
        # duration calculation - implied duration to next barline or song end
        if use_implied_duration:
            if i + 1 < len(filtered_barlines):
                duration = filtered_barlines[i + 1].time - barline.time
            else:
                duration = song_length - barline.time
        else:
            duration = 0
        
        barlines_encoded["duration"][i] = duration
        barlines_encoded["duration.s"][i] = 0  # placeholder, will be calculated later
        barlines_encoded["type"][i] = "Barline"
        subtype = barline.subtype.lower() if hasattr(barline, "subtype") and barline.subtype else ""
        barlines_encoded["value"][i] = check_text(text = f"{subtype}-barline" if subtype else "barline")
    
    return pd.DataFrame(data = barlines_encoded, columns = output_columns)

# scrape time signatures
def scrape_time_signatures(time_signatures, song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """
    Extract time signature changes with implied duration calculation.
    
    Processes time signature changes (ignoring the initial time signature) and calculates
    change ratios and durations until the next time signature change or song end.
    
    Parameters
    ----------
    time_signatures : list
        List of time signature objects with time, numerator, and denominator attributes
    song_length : int
        Total length of the song in time steps, used as end boundary for duration calculation
    use_implied_duration : bool, optional
        Whether to calculate implied durations between time signature changes, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing time signature change data
    """
    output_columns = EXPRESSION_TEXT_COLUMNS
    if len(time_signatures) <= 1:
        return pd.DataFrame(columns = output_columns)
    
    # we track changes, so ignore first time signature
    changes = time_signatures[1:]
    time_signatures_encoded = {key: [None] * len(changes) for key in output_columns}
    
    for i, time_signature in enumerate(changes):
        time_signatures_encoded["time"][i] = time_signature.time
        time_signatures_encoded["time.s"][i] = 0  # placeholder, will be calculated later
        
        # duration calculation - implied duration to next time signature change or song end
        if use_implied_duration:
            if i + 1 < len(changes):
                duration = changes[i + 1].time - time_signature.time
            else:
                duration = song_length - time_signature.time
        else:
            duration = 0
            
        time_signatures_encoded["duration"][i] = duration
        time_signatures_encoded["duration.s"][i] = 0  # placeholder, will be calculated later
        time_signatures_encoded["type"][i] = "TimeSignature"
        
        # calculate time signature change ratio
        prev_ts = time_signatures[i]  # previous time signature (since changes starts from index 1)
        if (hasattr(time_signature, "numerator") and hasattr(time_signature, "denominator") and
            hasattr(prev_ts, "numerator") and hasattr(prev_ts, "denominator") and
            time_signature.denominator and prev_ts.denominator):
            time_signature_change_ratio = (time_signature.numerator / time_signature.denominator) / (prev_ts.numerator / prev_ts.denominator)
        else:
            time_signature_change_ratio = 1.0
        
        time_signatures_encoded["value"][i] = check_text(text = time_signature_change_mapper(time_signature_change_ratio = time_signature_change_ratio))
    
    return pd.DataFrame(data = time_signatures_encoded, columns = output_columns)

# scrape key signatures
def scrape_key_signatures(key_signatures, song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """
    Extract key signature changes with implied duration calculation.
    
    Processes key signature changes (ignoring the initial key signature) and calculates
    circle-of-fifths distances and durations until the next key change or song end.
    
    Parameters
    ----------
    key_signatures : list
        List of key signature objects with time and fifths attributes
    song_length : int
        Total length of the song in time steps, used as end boundary for duration calculation
    use_implied_duration : bool, optional
        Whether to calculate implied durations between key signature changes, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing key signature change data
    """
    output_columns = EXPRESSION_TEXT_COLUMNS
    if len(key_signatures) <= 1:
        return pd.DataFrame(columns = output_columns)
    
    # we track changes, so ignore first key signature
    changes = key_signatures[1:]
    key_signatures_encoded = {key: [None] * len(changes) for key in output_columns}
    
    for i, key_signature in enumerate(changes):
        key_signatures_encoded["time"][i] = key_signature.time
        key_signatures_encoded["time.s"][i] = 0  # placeholder, will be calculated later
        
        # duration calculation - implied duration to next key signature change or song end
        if use_implied_duration:
            if i + 1 < len(changes):
                duration = changes[i + 1].time - key_signature.time
            else:
                duration = song_length - key_signature.time
        else:
            duration = 0
            
        key_signatures_encoded["duration"][i] = duration
        key_signatures_encoded["duration.s"][i] = 0  # placeholder, will be calculated later
        key_signatures_encoded["type"][i] = "KeySignature"
        
        # calculate key change distance
        prev_ks = key_signatures[i]  # previous key signature (since changes starts from index 1)
        if (hasattr(key_signature, "fifths") and hasattr(prev_ks, "fifths") and
            key_signature.fifths is not None and prev_ks.fifths is not None):
            distance = key_signature.fifths - prev_ks.fifths
            # calculate minimum distance considering circle of fifths
            if distance != 0:
                circular_distance = min(distance, (-(distance / abs(distance)) * 12) + distance, key = lambda dist: abs(dist))
            else:
                circular_distance = 0
        else:
            circular_distance = 0
        
        key_signatures_encoded["value"][i] = check_text(text = f"key-signature-change-{int(circular_distance)}")
    
    return pd.DataFrame(data = key_signatures_encoded, columns = output_columns)

# scrape tempos
def scrape_tempos(tempos, song_length: int, use_implied_duration: bool = True) -> pd.DataFrame:
    """
    Extract tempo markings with implied duration calculation.
    
    Processes tempo changes and maps QPM values to standard tempo markings,
    calculating durations until the next tempo change or song end.
    
    Parameters
    ----------
    tempos : list
        List of tempo objects with time and qpm attributes
    song_length : int
        Total length of the song in time steps, used as end boundary for duration calculation
    use_implied_duration : bool, optional
        Whether to calculate implied durations between tempo changes, by default True
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by EXPRESSION_TEXT_COLUMNS containing tempo data
    """
    output_columns = EXPRESSION_TEXT_COLUMNS
    if not tempos:
        return pd.DataFrame(columns = output_columns)
    
    tempos_encoded = {key: [None] * len(tempos) for key in output_columns}
    
    for i, tempo in enumerate(tempos):
        tempos_encoded["time"][i] = tempo.time
        tempos_encoded["time.s"][i] = 0  # placeholder, will be calculated later
        
        # duration calculation - implied duration to next tempo or song end
        if use_implied_duration:
            if i + 1 < len(tempos):
                duration = tempos[i + 1].time - tempo.time
            else:
                duration = song_length - tempo.time
        else:
            duration = 0
            
        tempos_encoded["duration"][i] = duration
        tempos_encoded["duration.s"][i] = 0  # placeholder, will be calculated later
        tempos_encoded["type"][i] = "Tempo"
        qpm = tempo.qpm if hasattr(tempo, "qpm") else 120
        tempos_encoded["value"][i] = check_text(text = qpm_tempo_mapper(qpm = qpm))
    
    return pd.DataFrame(data = tempos_encoded, columns = output_columns)

# scrape system-level expression text
def scrape_system_expression_text(music: muspy_express.Music, song_length: int) -> pd.DataFrame:
    """
    Extract system-level expression text from a music object.
    
    Combines annotations, barlines, time signatures, key signatures, and tempos
    from the system level into a single DataFrame of expression text.
    
    Parameters
    ----------
    music : muspy_express.Music
        Music object containing system-level musical elements
    song_length : int
        Total length of the song in time steps
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all system-level expression text
    """
    system_annotations = scrape_annotations(annotations = music.annotations, song_length = song_length) if hasattr(music, "annotations") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    system_barlines = scrape_barlines(barlines = music.barlines, song_length = song_length) if hasattr(music, "barlines") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    system_time_signatures = scrape_time_signatures(time_signatures = music.time_signatures, song_length = song_length) if hasattr(music, "time_signatures") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    system_key_signatures = scrape_key_signatures(key_signatures = music.key_signatures, song_length = song_length) if hasattr(music, "key_signatures") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    system_tempos = scrape_tempos(tempos = music.tempos, song_length = song_length) if hasattr(music, "tempos") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    
    system_level_expression_text = pd.concat(objs = (system_annotations, system_barlines, system_time_signatures, system_key_signatures, system_tempos), axis = 0, ignore_index = True)
    return system_level_expression_text

# scrape staff-level expression text
def scrape_staff_expression_text(track: muspy_express.Track, music: muspy_express.Music, song_length: int) -> pd.DataFrame:
    """
    Extract staff-level expression text from a single track.
    
    Processes track-specific annotations and other staff-level musical elements.
    Can be extended to include articulations, slurs, pedal markings, etc.
    
    Parameters
    ----------
    track : muspy_express.Track
        Individual track/staff containing musical elements
    music : muspy_express.Music
        Parent music object for context
    song_length : int
        Total length of the song in time steps
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing staff-level expression text
    """
    staff_annotations = scrape_annotations(annotations = track.annotations, song_length = song_length) if hasattr(track, "annotations") else pd.DataFrame(columns = EXPRESSION_TEXT_COLUMNS)
    # you could add more staff-level features here like articulations, slurs, pedals if needed
    return staff_annotations

##################################################


# MAIN EXTRACTOR FUNCTION
##################################################

def extract(music: muspy_express.Music) -> List[dict]:
    """
    extract expression text from a music object and return list of track dictionaries.
    
    Parameters
    ----------
    music : muspy_express.Music
        the music object to extract from
        
    Returns
    -------
    List[dict]
        list of track dictionaries containing expression text
    """
    
    # get song length using muspy2 API
    song_length = music.get_end_time()
    
    # extract system-level lyrics and expression text
    system_lyrics = scrape_lyrics(lyrics = music.lyrics)
    if len(system_lyrics) > 0:
        # muspy2 uses get_real_time instead of metrical_time_to_absolute_time
        if hasattr(music, "get_real_time"):
            system_lyrics["time.s"] = system_lyrics["time"].apply(lambda time: music.get_real_time(time))
            system_lyrics["duration.s"] = system_lyrics["duration"].apply(lambda duration: music.get_real_time(duration) if duration > 0 else 0.0)
        else:
            system_lyrics["time.s"] = system_lyrics["time"] * 0.5  # fallback estimate
            system_lyrics["duration.s"] = system_lyrics["duration"] * 0.5  # fallback estimate
        system_lyrics = system_lyrics[EXPRESSION_TEXT_COLUMNS]
    
    system_expression_text = scrape_system_expression_text(music = music, song_length = song_length)
    if len(system_expression_text) > 0:
        # muspy2 uses get_real_time instead of metrical_time_to_absolute_time
        if hasattr(music, "get_real_time"):
            system_expression_text["time.s"] = system_expression_text["time"].apply(lambda time: music.get_real_time(time))
            system_expression_text["duration.s"] = system_expression_text["duration"].apply(lambda duration: music.get_real_time(duration) if duration > 0 else 0.0)
        else:
            system_expression_text["time.s"] = system_expression_text["time"] * 0.5  # fallback estimate
            system_expression_text["duration.s"] = system_expression_text["duration"] * 0.5  # fallback estimate
        system_expression_text = system_expression_text[EXPRESSION_TEXT_COLUMNS]

    # list to store all track dictionaries for this song
    song_output = []

    # loop through tracks and extract expression text
    for track_idx, track in enumerate(music.tracks):
        
        # skip drum tracks or unknown programs
        if track.is_drum or track.program not in KNOWN_PROGRAMS:
            continue
        
        # extract track-level expression text
        track_expression_text = scrape_staff_expression_text(track = track, music = music, song_length = song_length)
        if len(track_expression_text) > 0:
            if hasattr(music, "get_real_time"):
                track_expression_text["time.s"] = track_expression_text["time"].apply(lambda time: music.get_real_time(time))
                track_expression_text["duration.s"] = track_expression_text["duration"].apply(lambda duration: music.get_real_time(duration) if duration > 0 else 0.0)
            else:
                track_expression_text["time.s"] = track_expression_text["time"] * 0.5
                track_expression_text["duration.s"] = track_expression_text["duration"] * 0.5
            track_expression_text = track_expression_text[EXPRESSION_TEXT_COLUMNS]
        
        # get staff-level lyrics
        staff_lyrics = scrape_lyrics(lyrics = track.lyrics)
        if len(staff_lyrics) > 0:
            if hasattr(music, "get_real_time"):
                staff_lyrics["time.s"] = staff_lyrics["time"].apply(lambda time: music.get_real_time(time))
                staff_lyrics["duration.s"] = staff_lyrics["duration"].apply(lambda duration: music.get_real_time(duration) if duration > 0 else 0.0)
            else:
                staff_lyrics["time.s"] = staff_lyrics["time"] * 0.5
                staff_lyrics["duration.s"] = staff_lyrics["duration"] * 0.5
            staff_lyrics = staff_lyrics[EXPRESSION_TEXT_COLUMNS]
        
        # combine all expression text
        expression_text = pd.concat(objs = (system_expression_text, system_lyrics, track_expression_text, staff_lyrics), axis = 0, ignore_index = True)

        # create output dictionary for this track
        track_output = {
            "track" : track_idx,
            "program" : track.program,
            "is_drum" : bool(track.is_drum),
            "resolution" : music.resolution,
            "track_length" : {
                "time_steps": song_length,
                "seconds": music.get_real_time(song_length) if hasattr(music, "get_real_time") else song_length * 0.5,
                "bars": len(music.barlines) if hasattr(music, "barlines") else 0,
                "beats": len(music.beats) if hasattr(music, "beats") else 0},
            "expression_text" : expression_text
        }

        # add this track to the song output (no threshold check)
        song_output.append(track_output)

    return song_output

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
        parser = argparse.ArgumentParser(prog = "Extract Expression Text from PDMX", description = "Extract expression text from PDMX.") # create argument parser
        parser.add_argument("--pdmx_filepath", type = str, default = PDMX_FILEPATH, help = "Path to PDMX file.")
        parser.add_argument("--musescore_dir", type = str, default = MUSESCORE_DIR, help = "Path to MuseScore directory.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Path to output directory.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.pdmx_filepath):
            raise FileNotFoundError(f"PDMX file not found: {args.pdmx_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # set up output directory
    print("Setting up output directory...")
    if exists(args.output_dir) and args.reset:
        rmtree(args.output_dir)
    if not exists(args.output_dir): # create output directory if it doesn't exist
        makedirs(args.output_dir)
    expression_pickles_dir = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(expression_pickles_dir):
        mkdir(expression_pickles_dir)
    output_filepath = f"{args.output_dir}/{DATASET_NAME}.csv"
    pdmx_preformatted_filepath = f"{args.output_dir}/pdmx_preformatted.csv"
    mscz_issues_filepath, mxl_issues_filepath = f"{args.output_dir}/mscz_issues.csv", f"{args.output_dir}/mxl_issues.csv"
    print("Set up output directory.")

    # create pdmx preformatted if necessary
    if not exists(pdmx_preformatted_filepath) or args.reset:

        # load pdmx
        print("PDMX preformatted not found, loading PDMX...")
        pdmx = pd.read_csv(filepath_or_buffer = args.pdmx_filepath, sep = ",", header = 0, index_col = False)
        pdmx_dir = dirname(args.pdmx_filepath)
        pdmx = pdmx.drop(columns = ["path", "metadata", "pdf", "mid"]) # drop everything but mxl
        pdmx = pdmx[pdmx["subset:all_valid"] & pdmx["subset:no_license_conflict"]] # use the correct subset, only valid pdmx paths
        pdmx = pdmx.drop(columns = list(filter(lambda column: column.startswith("subset:"), pdmx.columns))) # drop subset columns
        pdmx["mxl"] = pdmx["mxl"].apply(lambda mxl: f"{pdmx_dir}/{mxl[2:]}") # add pdmx_dir to mxl, converting to absolute path
        pdmx = pdmx.rename(columns = {"mxl": "path_mxl"}) # rename mxl to path
        pdmx = pdmx.reset_index(drop = True) # reset index
        output_columns = pdmx.columns.tolist()
        print("Loaded PDMX.")

        # get musescore paths
        print("Determining MuseScore paths...")
        musescore_paths = glob(f"{args.musescore_dir}/**/*.mscz", recursive = True)
        id_to_musescore_path = {id_: path for path, id_ in zip(musescore_paths, map(get_id_from_path, musescore_paths))}
        pdmx["path_mscz"] = pdmx["path_mxl"].apply(lambda path_mxl: id_to_musescore_path.get(get_id_from_path(path_mxl), None))
        output_columns.insert(output_columns.index("path_mxl") + 1, "path_mscz") # add to output columns
        del musescore_paths, id_to_musescore_path
        print("Determined MuseScore paths.")

        # get output paths
        print("Determining output paths...")
        pdmx["path_expression"] = [f"{expression_pickles_dir}/{basename(path_mxl)[:-len('.mxl')]}.pkl" for path_mxl in pdmx["path_mxl"]]
        output_columns.insert(0, "path_expression")
        print("Determined output paths.")

        # write pdmx preformatted
        print("Writing PDMX preformatted...")
        pdmx = pdmx[output_columns]
        pdmx.to_csv(path_or_buf = pdmx_preformatted_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        print("Wrote PDMX preformatted.")

        # free up memory
        del pdmx, output_columns

    # load pdmx preformatted
    print("Loading PDMX preformatted...")
    dataset = pd.read_csv(filepath_or_buffer = pdmx_preformatted_filepath, sep = ",", header = 0, index_col = False)
    print(f"Loaded PDMX preformatted with {len(dataset)} paths.")
    n_missing_mscz = sum(pd.isna(dataset["path_mscz"]))
    if n_missing_mscz > 0:
        print(f"There are {n_missing_mscz} paths with no MSCZ file (likely lost in data transfer).")
    del n_missing_mscz

    # write column names
    print("Writing column names...")
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = dataset.columns.tolist()).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been completed yet
        print(f"Wrote column names to {output_filepath}, no paths have been completed yet.")
    elif exists(output_filepath):
        already_completed_paths = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["path_expression"])["path_expression"]) # get already completed paths
        mscz_issues_paths = set(pd.read_csv(filepath_or_buffer = mscz_issues_filepath, sep = ",", header = 0, index_col = False, usecols = ["path"])["path"])
        mxl_issues_paths = set(pd.read_csv(filepath_or_buffer = mxl_issues_filepath, sep = ",", header = 0, index_col = False, usecols = ["path"])["path"])
        already_completed_paths.update(mscz_issues_paths.intersection(mxl_issues_paths)) # add paths that weren't written because they both threw errors
        del mscz_issues_paths, mxl_issues_paths
        print(f"Column names already written to {output_filepath}, {len(already_completed_paths)} paths have been completed.")

    # write issue file headers
    if not exists(mscz_issues_filepath) or args.reset:
        pd.DataFrame(columns = ISSUES_COLUMNS).to_csv(path_or_buf = mscz_issues_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        print(f"Wrote MuseScore issues column names to {mscz_issues_filepath}.")
    if not exists(mxl_issues_filepath) or args.reset:
        pd.DataFrame(columns = ISSUES_COLUMNS).to_csv(path_or_buf = mxl_issues_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
        print(f"Wrote MusicXML issues column names to {mxl_issues_filepath}.")

    # determining paths to complete
    print("Determining paths to complete...")
    indices_to_complete = [i for i, path_expression in enumerate(dataset["path_expression"]) if path_expression not in already_completed_paths]
    del already_completed_paths
    print(f"{len(indices_to_complete)} paths remaining to complete.")

    ##################################################


    # FUNCTION TO EXTRACT DATA
    ##################################################

    # silence warnings during file processing
    warnings.filterwarnings("ignore")
    
    # clean string for csv issues
    clean_issue_text = lambda text: str(text).replace("\n", " ").replace(",", " ")

    # helper function to extract data from a single PDMX entry
    def extract_helper(i: int):
        """
        Extract expression text from a single PDMX entry and save as pickle.
        
        Loads both MSCZ and MXL formats, extracts expression text using the core extract function,
        and saves the result as a pickle file containing a list of track dictionaries.
        
        Parameters
        ----------
        i : int
            Row index in the dataset DataFrame corresponding to the file to process
            
        Returns
        -------
        None
            Function saves results to disk and appends to CSV output file
        """

        # get row
        row = dataset.loc[i]
        path_expression = row["path_expression"]
        path_mscz = row["path_mscz"]
        path_mxl = row["path_mxl"]

        # read both file formats with error handling
        music_mxl = None
        music_mscz = None
        
        # read MXL file
        try:
            music_mxl = muspy_express.read_musicxml(path_mxl)
        except Exception as e:
            issue_row = pd.DataFrame({"path": [path_mxl], "issue": [clean_issue_text(text = str(e))]})
            issue_row.to_csv(path_or_buf = mxl_issues_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")
            print(f"Error reading MXL {basename(path_mxl)}: {e}")
        
        # read MSCZ file if available
        if not pd.isna(path_mscz):
            try:
                music_mscz = muspy_express.read_musescore(path_mscz)
            except Exception as e:
                issue_row = pd.DataFrame({"path": [path_mscz], "issue": [clean_issue_text(text = str(e))]})
                issue_row.to_csv(path_or_buf = mscz_issues_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")
                print(f"Error reading MSCZ {basename(path_mscz)}: {e}")
        
        # determine which music object to use (prefer MSCZ, fallback to MXL)
        music = music_mxl if music_mxl is not None else music_mscz
        
        # if both files failed to read, skip this entry
        if music is None:
            print(f"Both MXL and MSCZ failed to read for {basename(path_mxl)[:-len('.mxl')]}, skipping...")
            return

        # use the core extract function
        song_output = extract(music = music)
        
        # add file paths to each track dictionary
        for track_output in song_output:
            track_output["path_mscz"] = path_mscz
            track_output["path_mxl"] = path_mxl

        # save the list of track dictionaries as one pickle per song
        with open(path_expression, "wb") as pickle_file:
            pickle.dump(obj = song_output, file = pickle_file, protocol = pickle.HIGHEST_PROTOCOL)

        # write data row to CSV
        row.to_frame().T.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")

        # return nothing
        return

    ##################################################


    # EXTRACT DATA WITH MULTIPROCESSING
    ##################################################

    # use multiprocessing
    print("Extracting expression text...")
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
            func = extract_helper,
            iterable = indices_to_complete,
            chunksize = 1,
        ),
        desc = "Extracting",
        total = len(indices_to_complete)))
    print("Extracted expression text.")

    ##################################################

##################################################