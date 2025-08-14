# README
# Phillip Long
# August 9, 2025

# Test file for extract_expression_text.py
# Tests the end-to-end process of loading a music object and extracting expression text.

# IMPORTS
##################################################

import sys
import pickle
import pandas as pd
from os import remove
from os.path import dirname, realpath, abspath, exists
from pprint import pprint

# add paths for imports
sys.path.insert(0, dirname(dirname(abspath(__file__))))  # add muspy_express to path
sys.path.insert(0, "/home/pnlong")  # add path to muspy2

# import muspy_express (muspy2)
try:
    from muspy2 import muspy as muspy_express
    print("✓ successfully imported muspy2")
except ImportError as e:
    print(f"Error: muspy2 not found: {e}")
    print("Make sure you're running in conda base environment.")
    print("Run: conda activate base")
    sys.exit(1)

from extract_expression_text import extract

##################################################


# CONSTANTS
##################################################

# test file paths
TEST_MXL_PATH = "/deepfreeze/pnlong/PDMX/PDMX/mxl/0/0/Qma1qU8ywCZ8p9h3eoL2uR3Ld7sgZXSotgCszHZmFUkVR1.mxl"
OUTPUT_PICKLE_PATH = "/home/pnlong/muspy_express/workshop_plots/test_output.pkl"

##################################################


# HELPER FUNCTIONS
##################################################

def cleanup_test_files():
    """
    Clean up temporary test files created during testing.
    
    Removes the pickle output file if it exists to keep the workspace clean.
    
    Returns
    -------
    None
        Removes test files from disk
    """
    if exists(OUTPUT_PICKLE_PATH):
        try:
            remove(OUTPUT_PICKLE_PATH)
            print(f"   ✓ cleaned up test file: {OUTPUT_PICKLE_PATH}")
        except Exception as e:
            print(f"   ✗ failed to remove test file: {e}")

def pretty_print_output(song_output):
    """
    Pretty print the song output with proper formatting.
    
    Parameters
    ----------
    song_output : list
        List of track dictionaries from extract function
        
    Returns
    -------
    None
        Prints formatted output to console
    """
    print("\n" + "=" * 80)
    print("PRETTY PRINTED OUTPUT")
    print("=" * 80)
    
    print(f"\nSong contains {len(song_output)} tracks:")
    
    for i, track_dict in enumerate(song_output):
        print(f"\n--- TRACK {i} ---")
        print(f"Track index: {track_dict['track']}")
        print(f"Program: {track_dict['program']}")
        print(f"Is drum: {track_dict['is_drum']}")
        print(f"Resolution: {track_dict['resolution']}")
        
        print(f"Track length:")
        for key, value in track_dict["track_length"].items():
            print(f"  {key}: {value}")
        
        print(f"Expression text ({len(track_dict['expression_text'])} items):")
        if len(track_dict["expression_text"]) > 0:
            print(track_dict["expression_text"].to_string(index = False))
        else:
            print("  No expression text found")
        
        # add file paths if present
        if "path_mscz" in track_dict:
            print(f"Source MSCZ: {track_dict['path_mscz']}")
        if "path_mxl" in track_dict:
            print(f"Source MXL: {track_dict['path_mxl']}")

##################################################


# TEST FUNCTIONS
##################################################

def test_extract_expression_text():
    """
    Test the end-to-end process of extracting expression text from a music file.
    
    Loads a test MXL file, extracts expression text, saves to pickle, and verifies output.
    Also includes pretty printing of the complete output structure.
    
    Returns
    -------
    bool
        True if all tests pass, False otherwise
    """
    
    print("=" * 80)
    print("TESTING EXTRACT EXPRESSION TEXT")
    print("=" * 80)
    
    # load the music object
    print("\n1. loading music object...")
    print(f"   file: {TEST_MXL_PATH}")
    try:
        music = muspy_express.read_musicxml(TEST_MXL_PATH)
        print(f"   ✓ successfully loaded music object")
        print(f"   - song length: {music.get_end_time()} time steps")
        print(f"   - resolution: {music.resolution}")
        print(f"   - number of tracks: {len(music.tracks)}")
        
        # print track info
        for i, track in enumerate(music.tracks):
            print(f"   - track {i}: program {track.program}, is_drum={track.is_drum}, {len(track.notes)} notes")
            
    except Exception as e:
        print(f"   ✗ failed to load music object: {e}")
        return False
    
    # extract expression text
    print("\n2. extracting expression text...")
    try:
        song_output = extract(music = music)
        print(f"   ✓ successfully extracted expression text")
        print(f"   - number of valid tracks: {len(song_output)}")
        
        # print summary of each track
        for track_dict in song_output:
            track_idx = track_dict["track"]
            program = track_dict["program"]
            n_expression_items = len(track_dict["expression_text"])
            print(f"   - track {track_idx}: program {program}, {n_expression_items} expression text items")
            
    except Exception as e:
        print(f"   ✗ failed to extract expression text: {e}")
        return False
    
    # pretty print the complete output
    pretty_print_output(song_output)
    
    # save pickle output
    print("\n3. saving pickle output...")
    try:
        with open(OUTPUT_PICKLE_PATH, "wb") as pickle_file:
            pickle.dump(obj = song_output, file = pickle_file, protocol = pickle.HIGHEST_PROTOCOL)
        print(f"   ✓ successfully saved pickle to {OUTPUT_PICKLE_PATH}")
    except Exception as e:
        print(f"   ✗ failed to save pickle: {e}")
        return False
    
    # load and verify pickle
    print("\n4. loading and verifying pickle...")
    try:
        with open(OUTPUT_PICKLE_PATH, "rb") as pickle_file:
            loaded_output = pickle.load(pickle_file)
        print(f"   ✓ successfully loaded pickle")
        print(f"   - type: {type(loaded_output)}")
        print(f"   - length: {len(loaded_output)}")
        
        # verify structure
        if len(loaded_output) > 0:
            first_track = loaded_output[0]
            print(f"   - first track keys: {list(first_track.keys())}")
            print(f"   - expression text type: {type(first_track['expression_text'])}")
            print(f"   - expression text shape: {first_track['expression_text'].shape}")
            
            # show first few expression text items
            if len(first_track["expression_text"]) > 0:
                print(f"   - first few expression text items:")
                print(first_track["expression_text"].head().to_string(index = False))
            else:
                print(f"   - no expression text found")
                
    except Exception as e:
        print(f"   ✗ failed to load/verify pickle: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


def inspect_music_object():
    """
    Inspect the music object structure to understand its attributes and data.
    
    Useful for debugging and understanding the muspy_express Music and Track objects.
    
    Returns
    -------
    None
        Prints inspection results to console
    """
    
    print("\n" + "=" * 80)
    print("INSPECTING MUSIC OBJECT STRUCTURE")
    print("=" * 80)
    
    try:
        music = muspy_express.read_musicxml(TEST_MXL_PATH)
        
        print(f"\nmusic object attributes:")
        for attr in dir(music):
            if not attr.startswith("_"):
                try:
                    value = getattr(music, attr)
                    if callable(value):
                        print(f"  {attr}(): {type(value)}")
                    else:
                        print(f"  {attr}: {type(value)} = {value if len(str(value)) < 100 else str(value)[:100] + '...'}")
                except:
                    print(f"  {attr}: <could not access>")
        
        print(f"\ntrack 0 attributes:")
        if len(music.tracks) > 0:
            track = music.tracks[0]
            for attr in dir(track):
                if not attr.startswith("_"):
                    try:
                        value = getattr(track, attr)
                        if callable(value):
                            print(f"  {attr}(): {type(value)}")
                        else:
                            print(f"  {attr}: {type(value)} = {value if len(str(value)) < 100 else str(value)[:100] + '...'}")
                    except:
                        print(f"  {attr}: <could not access>")
                        
    except Exception as e:
        print(f"failed to inspect music object: {e}")

##################################################


# MAIN EXECUTION
##################################################

if __name__ == "__main__":
    """
    Main execution block for running the test suite.
    
    Runs the expression text extraction test, inspects the music object structure,
    and cleans up temporary files before exiting.
    """
    
    # run the main test
    success = test_extract_expression_text()
    
    # optionally inspect the music object structure
    inspect_music_object()
    
    # clean up temporary test files
    print("\n5. cleaning up test files...")
    cleanup_test_files()
    
    # exit with appropriate code
    sys.exit(0 if success else 1)

##################################################