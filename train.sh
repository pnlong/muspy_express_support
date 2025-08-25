#!/bin/bash

# README
# Phillip Long
# January 4, 2023

# A place to store the command-line inputs for different models to train.

# sh /home/pnlong/muspy_express/train.sh

# CONSTANTS
##################################################

# software filepaths
software_dir="/home/pnlong/muspy_express"
software="${software_dir}/train.py"

# defaults
data_dir="/deepfreeze/pnlong/muspy_express/experiments/metrical"
gpu=-1 # gpu number
unidimensional=""
resume=""
training_number=0 # which training to run (0-6)

# small model architecture, the default
dim=512 # dimension
layers=6 # layers
heads=8 # attention heads

##################################################


# COMMAND LINE ARGS
##################################################

# parse command line arguments
usage="Usage: $(basename ${0}) [-d] (data directory) [-u] (unidimensional?) [-r] (resume?) [-sml] (small/medium/large) [-g] (gpu to use) [-n] (training number 0-6)"
while getopts ':d:g:n:ursmlh' opt; do
  case "${opt}" in
    d) # also implies metrical/absolute time
      data_dir="${OPTARG}"
      ;;
    u) # unidimensional flag
      unidimensional="--unidimensional"
      ;;
    r) # whether to resume runs
      resume="--resume -1"
      ;;
    s) # small
      dim=512 # dimension
      layers=6 # layers
      heads=8 # attention heads
      ;;
    m) # medium
      dim=768 # dimension
      layers=10 # layers
      heads=8 # attention heads
      ;;
    l) # large
      dim=960 # dimension
      layers=12 # layers
      heads=12 # attention heads
      ;;
    g) # gpu to use
      gpu="${OPTARG}"
      ;;
    n) # training number to run
      training_number="${OPTARG}"
      ;;
    h)
      echo ${usage}
      exit 0
      ;;
    :)
      echo -e "option requires an argument.\n${usage}"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\n${usage}"
      exit 1
      ;;
  esac
done

paths_train="${data_dir}/train.txt"
paths_valid="${data_dir}/valid.txt"
encoding="${data_dir}/encoding.json"
output_dir="${data_dir}/models"

# constants
batch_size=8 # decrease if gpu memory consumption is too high
steps=80000 # in my experience >70000 is sufficient to train
sigma=8 # for anticipation, in seconds or beats depending on which time scale we are using

##################################################


# TRAIN MODELS
##################################################

set -e # stop if there's an error

if [ "${training_number}" -eq 0 ]; then # baseline
  python ${software} --baseline --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 1 ]; then # prefix, conditional on expressive features
  python ${software} --conditioning "prefix" --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 2 ]; then # anticipation, conditional on expressive features
  python ${software} --conditioning "anticipation" --sigma ${sigma} --conditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 3 ]; then # prefix, conditional on notes
  python ${software} --conditioning "prefix" --econditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 4 ]; then # anticipation, conditional on notes
  python ${software} --conditioning "anticipation" --sigma ${sigma} --econditional --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 5 ]; then # prefix, not conditional
  python ${software} --conditioning "prefix" --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
elif [ "${training_number}" -eq 6 ]; then # anticipation, not conditional
  python ${software} --conditioning "anticipation" --sigma ${sigma} --aug --paths_train ${paths_train} --paths_valid ${paths_valid} --encoding ${encoding} --output_dir ${output_dir} --batch_size ${batch_size} --steps ${steps} --dim ${dim} --layers ${layers} --heads ${heads} --gpu ${gpu} ${unidimensional} ${resume}
else
  echo "Error: Training number must be 0-6. Provided: ${training_number}"
  exit 1
fi

##################################################
