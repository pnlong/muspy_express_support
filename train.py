# README
# Phillip Long
# November 25, 2023

# Train a neural network.

# python /home/pnlong/model_musescore/train.py

# Absolute positional embedding (APE):
# python /home/pnlong/model_musescore/train.py

# Relative positional embedding (RPE):
# python /home/pnlong/model_musescore/train.py --no-abs_pos_emb --rel_pos_emb

# No positional embedding (NPE):
# python /home/pnlong/model_musescore/train.py --no-abs_pos_emb --no-rel_pos_emb


# IMPORTS
##################################################

import argparse
import logging
from os import makedirs, mkdir, environ
from os.path import exists, basename, dirname
import pprint
# import shutil # for copying files
import sys
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from typing import Dict
from tqdm import tqdm
import wandb
import datetime # for creating wandb run names linked to time of run

from dataset import MusicDataset
import music_x_transformers
import representation
from representation import ENCODING_FILEPATH
import encode
import utils

##################################################


# CONSTANTS
##################################################

DATA_DIR = "/home/pnlong/musescore/datav"
OUTPUT_DIR = "/home/pnlong/musescore/datav"
PROJECT_NAME = "ExpressionNet-Train"
INFER_RUN_NAME_STRING = "-1"

DEFAULT_MAX_SEQ_LEN = 1024

NA_VALUE = "NA"
ALL_STRING = "total"

PARTITIONS = ("train", "valid", "test")
RELEVANT_PARTITIONS = PARTITIONS[:2]
MASKS = (ALL_STRING, "note", "expressive") # for determining loss in notes vs expressive features
PERFORMANCE_METRICS = ("loss", "accuracy")
PERFORMANCE_OUTPUT_COLUMNS = ["step", "metric", "partition", "mask", "field", "value"]

PATHS_TRAIN = f"{DATA_DIR}/{RELEVANT_PARTITIONS[0]}.txt"
PATHS_VALID = f"{DATA_DIR}/{RELEVANT_PARTITIONS[1]}.txt"

# environment constants
# environ["CUDA_LAUNCH_BLOCKING"] = "1" # to ignore https://github.com/facebookresearch/detectron2/issues/2837 error (device-side assert triggered)
environ["WANDB_SILENT"] = "true" # to silence wandb outputs

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("-pt", "--paths_train", default = PATHS_TRAIN, type = str, help = ".txt file with absolute filepaths to training dataset.")
    parser.add_argument("-pv", "--paths_valid", default = PATHS_VALID, type = str, help = ".txt file with absolute filepaths to validation dataset.")
    parser.add_argument("-e", "--encoding", default = ENCODING_FILEPATH, type = str, help = ".json file with encoding information.")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory that contains any necessary files/subdirectories (such as model checkpoints) created at runtime")
    # data
    parser.add_argument("-bs", "--batch_size", default = 8, type = int, help = "Batch size")
    parser.add_argument("--aug", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use data augmentation")
    parser.add_argument("-c", "--conditioning", default = encode.DEFAULT_CONDITIONING, choices = encode.CONDITIONINGS, type = str, help = "Conditioning type")
    parser.add_argument("-s", "--sigma", default = encode.SIGMA, type = float, help = "Sigma anticipation value (for anticipation conditioning, ignored when --conditioning != 'anticipation')")
    parser.add_argument("--baseline", action = "store_true", help = "Whether or not this is training the baseline model. The baseline ignores all expressive features.")
    # model
    parser.add_argument("--max_seq_len", default = DEFAULT_MAX_SEQ_LEN, type = int, help = "Maximum sequence length")
    parser.add_argument("--dim", default = 512, type = int, help = "Model dimension")
    parser.add_argument("-l", "--layers", default = 6, type = int, help = "Number of layers")
    parser.add_argument("--heads", default = 8, type = int, help = "Number of attention heads")
    parser.add_argument("--dropout", default = 0.2, type = float, help = "Dropout rate")
    parser.add_argument("--abs_pos_emb", action = argparse.BooleanOptionalAction, default = True, help = "Whether to use absolute positional embedding")
    parser.add_argument("--rel_pos_emb", action = argparse.BooleanOptionalAction, default = False, help = "Whether to use relative positional embedding")
    parser.add_argument("--conditional", action = "store_true", help = "Do we want to train in a purely conditional way, masking out the loss on expressive-feature tokens?")
    parser.add_argument("--econditional", action = "store_true", help = "Do we want to train in a purely conditional way, masking out the loss on note tokens?")
    parser.add_argument("--unidimensional", action = "store_true", help = "Should we train a model with unidimensional, as opposed to multidimensional, inputs?")
    # training
    parser.add_argument("--steps", default = 100000, type = int, help = "Number of steps")
    parser.add_argument("--valid_steps", default = 1000, type = int, help = "Validation frequency")
    parser.add_argument("--early_stopping", action = argparse.BooleanOptionalAction, default = False, help = "Whether to use early stopping")
    parser.add_argument("--early_stopping_tolerance", default = 20, type = int, help = "Number of extra validation rounds before early stopping")
    parser.add_argument("-lr", "--learning_rate", default = 0.0005, type = float, help = "Learning rate")
    parser.add_argument("--lr_warmup_steps", default = 5000, type = int, help = "Learning rate warmup steps")
    parser.add_argument("--lr_decay_steps", default = 100000, type = int, help = "Learning rate decay end steps")
    parser.add_argument("--lr_decay_multiplier", default = 0.1, type = float, help = "Learning rate multiplier at the end")
    parser.add_argument("--grad_norm_clip", default = 1.0, type = float, help = "Gradient norm clipping")
    # others
    parser.add_argument("-r", "--resume", default = None, type = str, help = "Provide the wandb run name/id to resume a run")
    parser.add_argument("-g", "--gpu", default = -1, type = int, help = "GPU number")
    parser.add_argument("-j", "--jobs", default = 4, type = int, help = "Number of workers for data loading")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# HELPER FUNCTIONS FOR TRAINING
##################################################

def get_lr_multiplier(step: int, warmup_steps: int, decay_end_steps: int, decay_end_multiplier: float) -> float:
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def get_accuracies(model_output: torch.tensor, seq: torch.tensor, unidimensional: bool = False, n_tokens_per_event: int = 1) -> torch.tensor:
    """Get the matrix of accuracies for correctly predicted tokens."""
    if unidimensional:
        predicted = torch.argmax(input = model_output, dim = -1, keepdim = False)
        expected = seq[:, n_tokens_per_event:]
    else:
        # predicted = torch.argmax(input = model_output, dim = -1, keepdim = False).transpose(dim0 = 0, dim1 = 1).transpose(dim0 = 1, dim1 = 2) # would work for constant vocabulary size
        predicted = torch.cat(tensors = [torch.argmax(input = output_field, dim = -1, keepdim = True) for output_field in model_output], dim = -1) # convert model output into a matrix of (greedy) predictions
        expected = seq[:, n_tokens_per_event:, :]
    return torch.eq(input = predicted, other = expected) # compare predicted to expected values

def generate_note_expressive_mask(seq: torch.tensor, encoding: dict = representation.get_encoding(), unidimensional: bool = False, n_tokens_per_event: int = 1) -> Dict[str, torch.tensor]:
    """Generate both a note and expressive-feature mask over a sequence, return as a dictionary using MASKS as keys."""
    event_types = seq[:, (n_tokens_per_event + encoding["unidimensional_encoding_order"].index("type"))::n_tokens_per_event] if unidimensional else seq[:, n_tokens_per_event:, encoding["dimensions"].index("type")]
    mask = {
        MASKS[0]: torch.repeat_interleave(input = torch.ones_like(input = event_types, dtype = torch.bool).to(device), repeats = n_tokens_per_event, dim = -1), # no mask
        MASKS[1]: torch.repeat_interleave(input = torch.logical_or(input = (event_types == encoding["type_code_map"]["note"]), other = (event_types == encoding["type_code_map"]["grace-note"])), repeats = n_tokens_per_event, dim = -1), # mask_note is true if the type is a note
        MASKS[2]: torch.repeat_interleave(input = (event_types == encoding["type_code_map"][representation.EXPRESSIVE_FEATURE_TYPE_STRING]), repeats = n_tokens_per_event, dim = -1) # mask_expressive is true if the type is an expressive feature
    }
    return mask

def calculate_loss_statistics(losses: torch.tensor, mask: torch.tensor = None, unidimensional: bool = False, n_tokens_per_event: int = 1) -> Dict[str, float]:
    """
    Calculate total loss and loss per field.
    """

    # apply mask
    if unidimensional:
        losses = losses.reshape(losses.shape[0], int(losses.shape[1] / n_tokens_per_event), n_tokens_per_event)
        mask = mask.reshape(mask.shape[0], int(mask.shape[1] / n_tokens_per_event), n_tokens_per_event)
    else:
        mask = torch.repeat_interleave(input = mask.unsqueeze(dim = -1), repeats = losses.shape[-1], dim = -1)
    mask = mask.byte()
    losses *= mask # just perform a normal masking operation

    # loss by field
    losses_field = torch.sum(input = losses, dim = list(range(len(losses.shape) - 1))).nan_to_num(nan = 0.0) # get sum of losses
    losses_field /= torch.sum(input = mask.byte(), dim = list(range(len(mask.shape) - 1))) # average losses
    losses_field = losses_field.tolist() # convert to python list

    # loss
    loss = sum(losses_field)

    # return dictionary with total loss, then the loss by field
    return dict({ALL_STRING: loss}, **dict(zip(encoding["unidimensional_encoding_order" if unidimensional else "dimensions"], losses_field)))

def calculate_accuracy_statistics(accuracies: torch.tensor, mask: torch.tensor = None, unidimensional: bool = False, n_tokens_per_event: int = 1) -> Dict[str, float]:
    """
    Calculate total accuracy and accuracy per field.
    Well, this doesn't actually calculate that. This actually tallies up the number of tokens correct, which will be divided later-on to get actual accuracy.
    """

    # apply mask
    if unidimensional:
        accuracies = accuracies.reshape(accuracies.shape[0], int(accuracies.shape[1] / n_tokens_per_event), n_tokens_per_event)
        mask = mask.reshape(mask.shape[0], int(mask.shape[1] / n_tokens_per_event), n_tokens_per_event)
    else:
        mask = torch.repeat_interleave(input = mask.unsqueeze(dim = -1), repeats = accuracies.shape[-1], dim = -1)
    mask = mask.byte()
    accuracies = (accuracies.byte() * mask).bool() # just perform a normal masking operation
    
    # accuracy by field
    accuracies_field = torch.sum(input = accuracies, dim = list(range(len(accuracies.shape) - 1))).nan_to_num(nan = 0.0).tolist() # replace nan with 0 to ease future calculations

    # accuracy
    accuracy = torch.sum(input = torch.all(input = accuracies, dim = -1), dim = None).item()

    # return dictionary with total accuracy, then the accuracy by field
    return dict({ALL_STRING: accuracy}, **dict(zip(encoding["unidimensional_encoding_order" if unidimensional else "dimensions"], accuracies_field)))

##################################################


# MAIN FUNCTION
##################################################

if __name__ == "__main__":

    # LOAD UP MODEL
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # check filepath arguments
    if not exists(args.paths_train):
        raise ValueError("Invalid --paths_train argument. File does not exist.")
    if not exists(args.paths_valid):
        raise ValueError("Invalid --paths_valid argument. File does not exist.")
    run_name = args.resume # get runname
    args.resume = (run_name != None) # convert to boolean value
    
    # get the specified device
    device = torch.device(f"cuda:{abs(args.gpu)}" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")
    print(f"Using device: {device}")

    # load the encoding
    encoding = representation.load_encoding(filepath = args.encoding) if exists(args.encoding) else representation.get_encoding()

    # create the dataset and data loader
    print(f"Creating the data loader...")
    dataset = {
        RELEVANT_PARTITIONS[0]: MusicDataset(paths = args.paths_train, encoding = encoding, conditioning = args.conditioning, controls_are_notes = args.econditional, sigma = args.sigma, is_baseline = args.baseline, max_seq_len = args.max_seq_len, use_augmentation = args.aug, unidimensional = args.unidimensional),
        RELEVANT_PARTITIONS[1]: MusicDataset(paths = args.paths_valid, encoding = encoding, conditioning = args.conditioning, controls_are_notes = args.econditional, sigma = args.sigma, is_baseline = args.baseline, max_seq_len = args.max_seq_len, use_augmentation = args.aug, unidimensional = args.unidimensional)
        }
    data_loader = {
        RELEVANT_PARTITIONS[0]: torch.utils.data.DataLoader(dataset = dataset[RELEVANT_PARTITIONS[0]], batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset[RELEVANT_PARTITIONS[0]].collate),
        RELEVANT_PARTITIONS[1]: torch.utils.data.DataLoader(dataset = dataset[RELEVANT_PARTITIONS[1]], batch_size = args.batch_size, shuffle = True, num_workers = args.jobs, collate_fn = dataset[RELEVANT_PARTITIONS[1]].collate)
    }

    # create the model
    print(f"Creating model...")
    use_absolute_time = encoding["use_absolute_time"]
    model = music_x_transformers.MusicXTransformer(
        dim = args.dim,
        encoding = encoding,
        depth = args.layers,
        heads = args.heads,
        max_seq_len = args.max_seq_len,
        max_temporal = encoding["max_" + ("time" if use_absolute_time else "beat")],
        rotary_pos_emb = args.rel_pos_emb,
        use_abs_pos_emb = args.abs_pos_emb,
        embedding_dropout = args.dropout,
        attention_dropout = args.dropout,
        ff_dropout = args.dropout,
        unidimensional = args.unidimensional
    ).to(device)
    # kwargs = {"depth": args.layers, "heads": args.heads, "max_seq_len": args.max_seq_len, "max_temporal": encoding["max_beat"], "rotary_pos_emb": args.rel_pos_emb, "use_abs_pos_emb": args.abs_pos_emb, "emb_dropout": args.dropout, "attn_dropout": args.dropout, "ff_dropout": args.dropout} # for debugging
    n_parameters = sum(p.numel() for p in model.parameters()) # statistics
    n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) # statistics (model size)

    # determine the output directory based on arguments
    if args.abs_pos_emb:
        positional_embedding = "ape"
    elif args.rel_pos_emb:
        positional_embedding = "rpe"
    else:
        positional_embedding = "npe"
    model_size = int(n_parameters_trainable / 1e+6)
    if args.conditional and args.econditional: # default to conditional on notes if there are two conditional boolean arguments supplied
        args.econditional = False
    args.output_dir = args.output_dir + "/" + (args.conditioning if not args.baseline else "baseline") + ("_conditional" if args.conditional else "") + ("_econditional" if args.econditional else "") + ("_unidimensional" if args.unidimensional else "") + f"_{positional_embedding}_{model_size}M" # custom output directory based on arguments
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    CHECKPOINTS_DIR = f"{args.output_dir}/checkpoints" # models will be stored in the output directory
    if not exists(CHECKPOINTS_DIR):
        mkdir(CHECKPOINTS_DIR)

    # start a new wandb run to track the script
    group_name = ("absolute" if use_absolute_time else "metrical") + ("-unidimensional" if args.unidimensional else "") # basename(dirname(args.output_dir))
    if (run_name == INFER_RUN_NAME_STRING):
        run_name = next(filter(lambda name: name.startswith(basename(args.output_dir)), (run.name for run in wandb.Api().runs(f"philly/{PROJECT_NAME}", filters = {"group": group_name}))), None) # try to infer the run name
        args.resume = (run_name != None) # redefine args.resume in the event that no run name was supplied, but we can't infer one either
    if (run_name is None): # in the event we need to create a new run name
        current_datetime = datetime.datetime.now().strftime("%m%d%y%H%M")
        run_name = f"{basename(args.output_dir)}-{current_datetime}"
    run = wandb.init(config = dict(vars(args), **{"n_parameters": n_parameters, "n_parameters_trainable": n_parameters_trainable}), resume = "allow", project = PROJECT_NAME, group = group_name, name = run_name, id = run_name) # set project title, configure with hyperparameters

    # set up the logger
    logging_output_filepath = f"{args.output_dir}/train.log"
    log_hyperparameters = not (args.resume and exists(logging_output_filepath))
    logging.basicConfig(level = logging.INFO, format = "%(message)s", handlers = [logging.FileHandler(filename = logging_output_filepath, mode = "a" if args.resume else "w"), logging.StreamHandler(stream = sys.stdout)])

    # log command called and arguments, save arguments
    if log_hyperparameters:
        logging.info(f"Running command: python {' '.join(sys.argv)}")
        logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")
        args_output_filepath = f"{args.output_dir}/train_args.json"
        logging.info(f"Saved arguments to {args_output_filepath}")
        utils.save_args(filepath = args_output_filepath, args = args)
        del args_output_filepath # clear up memory
    else: # print previous loggings to stdout
        with open(logging_output_filepath, "r") as logging_output:
            print(logging_output.read())

    # load previous model and summarize if needed
    best_model_filepath = {partition: f"{CHECKPOINTS_DIR}/best_model.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    model_previously_created = args.resume and all(exists(filepath) for filepath in best_model_filepath.values())
    if model_previously_created:
        model.load_state_dict(torch.load(f = best_model_filepath[RELEVANT_PARTITIONS[1]]))
    else:
        logging.info(f"Number of parameters: {n_parameters:,}")
        logging.info(f"Number of trainable parameters: {n_parameters_trainable:,}")
    
    # create the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    best_optimizer_filepath = {partition: f"{CHECKPOINTS_DIR}/best_optimizer.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    if args.resume and all(exists(filepath) for filepath in best_optimizer_filepath.values()):
        optimizer.load_state_dict(torch.load(f = best_optimizer_filepath[RELEVANT_PARTITIONS[1]]))

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda step: get_lr_multiplier(step = step, warmup_steps = args.lr_warmup_steps, decay_end_steps = args.lr_decay_steps, decay_end_multiplier = args.lr_decay_multiplier))
    best_scheduler_filepath = {partition: f"{CHECKPOINTS_DIR}/best_scheduler.{partition}.pth" for partition in RELEVANT_PARTITIONS}
    if args.resume and all(exists(filepath) for filepath in best_scheduler_filepath.values()):
        scheduler.load_state_dict(torch.load(f = best_scheduler_filepath[RELEVANT_PARTITIONS[1]]))
    
    # get conditional mask type
    conditional_mask_type = 0
    if args.conditional:
        conditional_mask_type = 1
    elif args.econditional:
        conditional_mask_type = 2
    conditional_mask_type = MASKS[conditional_mask_type] # get conditional mask type

    ##################################################


    # TRAINING PROCESS
    ##################################################

    # create a file to record performance metrics
    output_filepath = f"{args.output_dir}/performance.csv"
    performance_columns_must_be_written = not (exists(output_filepath) and args.resume) # whether or not to write column names
    if performance_columns_must_be_written: # if column names need to be written
        pd.DataFrame(columns = PERFORMANCE_OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = True, index = False, mode = "w")
    full_fields = [ALL_STRING] + list(encoding["dimensions"])

    # initialize variables
    step = 0
    min_loss = {partition: float("inf") for partition in RELEVANT_PARTITIONS}
    if not performance_columns_must_be_written:
        previous_performance = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", na_values = NA_VALUE, header = 0, index_col = False) # read in previous performance values
        if len(previous_performance) > 0:
            for partition in RELEVANT_PARTITIONS:
                min_loss[partition] = float(previous_performance[(previous_performance["metric"] == PERFORMANCE_METRICS[0]) & (previous_performance["partition"] == partition) & (previous_performance["mask"] == ALL_STRING) & (previous_performance["field"] == ALL_STRING)]["value"].min(axis = 0)) # get minimum loss
            step = int(previous_performance["step"].max(axis = 0)) # update step
        del previous_performance
    if args.early_stopping:
        count_early_stopping = 0

    # print current step
    print(f"Current Step: {step:,}")

    # iterate for the specified number of steps
    train_iterator = iter(data_loader[RELEVANT_PARTITIONS[0]])
    while step < args.steps:

        # to store loss/accuracy values
        performance = {metric: {partition: {mask_type: {field: 0.0 for field in full_fields} for mask_type in MASKS} for partition in RELEVANT_PARTITIONS} for metric in PERFORMANCE_METRICS}

        # TRAIN
        ##################################################

        logging.info(f"Training...")

        model.train()
        count, count_token = 0, {mask_type: 0 for mask_type in MASKS}
        recent_metrics = {
            PERFORMANCE_METRICS[0]: np.empty(shape = (0,)),
            PERFORMANCE_METRICS[1]: np.empty(shape = (0, 2))
        }
        for batch in (progress_bar := tqdm(iterable = range(args.valid_steps), desc = "Training")):

            # get next batch
            try:
                batch = next(train_iterator)
            except (StopIteration):
                train_iterator = iter(data_loader[RELEVANT_PARTITIONS[0]]) # reinitialize dataset iterator
                batch = next(train_iterator)

            # get input and output pair
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            # calculate loss for the batch
            optimizer.zero_grad()
            masks = generate_note_expressive_mask(seq = seq, encoding = encoding, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) # determine mask for notes vs expressive features
            loss_batch, losses_batch, output_batch = model(
                seq = seq,
                mask = mask,
                return_list = True,
                reduce = False, # reduce = False so that we have loss at each token
                return_output = True,
                conditional_mask = masks[conditional_mask_type] # mask if a conditional model
            )
            accuracies_batch = get_accuracies(model_output = output_batch, seq = seq, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) # calculate accuracy

            # update parameters according to loss
            loss_batch.backward() # calculate gradients
            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = args.grad_norm_clip)
            optimizer.step() # update parameters
            scheduler.step() # update scheduler

            # compute the moving average of the loss
            recent_metrics[PERFORMANCE_METRICS[0]] = np.append(arr = recent_metrics[PERFORMANCE_METRICS[0]], values = [float(loss_batch)], axis = 0) # float(loss_batch) because it has a gradient attribute
            if len(recent_metrics[PERFORMANCE_METRICS[0]]) > 10:
                recent_metrics[PERFORMANCE_METRICS[0]] = np.delete(arr = recent_metrics[PERFORMANCE_METRICS[0]], obj = 0, axis = 0)
            loss_batch = np.mean(a = recent_metrics[PERFORMANCE_METRICS[0]], axis = 0)
            
            # compute the moving average of the accuracy
            accuracy_batch = accuracies_batch.reshape(accuracies_batch.shape[0], int(accuracies_batch.shape[1] / model.decoder.net.n_tokens_per_event), model.decoder.net.n_tokens_per_event) if args.unidimensional else accuracies_batch # reshape if necessary
            accuracy_batch = np.array(object = (torch.sum(input = torch.all(input = accuracy_batch, dim = -1), dim = None).item(), utils.product(l = accuracy_batch.shape[:-1])), dtype = np.float64) # accuracy_batch is a tuple with the number of correct and the number total
            recent_metrics[PERFORMANCE_METRICS[1]] = np.append(arr = recent_metrics[PERFORMANCE_METRICS[1]], values = accuracy_batch.reshape((1, -1)), axis = 0)
            if len(recent_metrics[PERFORMANCE_METRICS[1]]) > 10:
                recent_metrics[PERFORMANCE_METRICS[1]] = np.delete(arr = recent_metrics[PERFORMANCE_METRICS[1]], obj = 0, axis = 0)
            accuracy_batch = np.sum(a = recent_metrics[PERFORMANCE_METRICS[1]], axis = 0) # take sum across <=10 most recent different batches
            accuracy_batch = accuracy_batch[0] / accuracy_batch[1] # calculate accuracy (total correct / total count)

            # set progress bar
            progress_bar.set_postfix(loss = f"{loss_batch:8.4f}", accuracy = f"{100 * accuracy_batch:5.2f}%")

            # log training loss/accuracy for wandb
            wandb.log({f"{RELEVANT_PARTITIONS[0]}/{PERFORMANCE_METRICS[0]}": loss_batch, f"{RELEVANT_PARTITIONS[0]}/{PERFORMANCE_METRICS[1]}": accuracy_batch}, step = step)

            # update counts
            count += len(batch)
            for mask_type in MASKS:
                count_token[mask_type] += torch.sum(input = masks[mask_type], dim = None).item()

            # calculate losses and accuracy for different facets
            performance_batch = {
                PERFORMANCE_METRICS[0]: {mask_type: calculate_loss_statistics(losses = losses_batch, mask = mask_, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) for mask_type, mask_ in masks.items()},
                PERFORMANCE_METRICS[1]: {mask_type: calculate_accuracy_statistics(accuracies = accuracies_batch, mask = mask_, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) for mask_type, mask_ in masks.items()}
            }
            for metric in PERFORMANCE_METRICS:
                for mask_type in MASKS:
                    for field in full_fields:
                        performance[metric][RELEVANT_PARTITIONS[0]][mask_type][field] += performance_batch[metric][mask_type][field]

            # increment step
            step += 1

        # release GPU memory right away
        del seq, mask, masks, loss_batch, losses_batch, output_batch, accuracies_batch

        # compute average loss/accuracy across batches
        for mask_type in MASKS:
            for field in full_fields:
                performance[PERFORMANCE_METRICS[0]][RELEVANT_PARTITIONS[0]][mask_type][field] /= count # loss
                performance[PERFORMANCE_METRICS[1]][RELEVANT_PARTITIONS[0]][mask_type][field] /= (count_token[mask_type] if count_token[mask_type] > 0 else 1) # accuracy

        # log train info for wandb
        for metric in PERFORMANCE_METRICS:
            for mask_type in MASKS:
                wandb.log({f"{RELEVANT_PARTITIONS[0]}/{metric}/{mask_type}/{field}": value for field, value in performance[metric][RELEVANT_PARTITIONS[0]][mask_type].items()}, step = step)

        ##################################################


        # VALIDATE
        ##################################################

        logging.info(f"Validating...")

        model.eval()
        with torch.no_grad():

            count, count_token = 0, {mask_type: 0 for mask_type in MASKS} # counts
            for batch in tqdm(iterable = data_loader[RELEVANT_PARTITIONS[1]], desc = "Validating"):

                # get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)

                # pass through the model
                masks = generate_note_expressive_mask(seq = seq, encoding = encoding, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) # determine mask for notes vs expressive features
                loss_batch, losses_batch, output_batch = model(
                    seq = seq,
                    mask = mask,
                    return_list = True,
                    reduce = False, # reduce = False so that we have loss at each token
                    return_output = True,
                    conditional_mask = masks[conditional_mask_type] # mask if a conditional model
                )
                accuracies_batch = get_accuracies(model_output = output_batch, seq = seq, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) # calculate accuracies
                                
                # update counts
                count += len(batch)
                for mask_type in MASKS:
                    count_token[mask_type] += torch.sum(input = masks[mask_type], dim = None).item()
                
                # calculate losses and accuracy for different facets
                performance_batch = {
                    PERFORMANCE_METRICS[0]: {mask_type: calculate_loss_statistics(losses = losses_batch, mask = mask_, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) for mask_type, mask_ in masks.items()},
                    PERFORMANCE_METRICS[1]: {mask_type: calculate_accuracy_statistics(accuracies = accuracies_batch, mask = mask_, unidimensional = args.unidimensional, n_tokens_per_event = model.decoder.net.n_tokens_per_event) for mask_type, mask_ in masks.items()}
                }
                for metric in PERFORMANCE_METRICS:
                    for mask_type in MASKS:
                        for field in full_fields:
                            performance[metric][RELEVANT_PARTITIONS[1]][mask_type][field] += performance_batch[metric][mask_type][field]
                
        # release GPU memory right away
        del seq, mask, masks, loss_batch, losses_batch, output_batch, accuracies_batch
        
        # compute average loss/accuracy across batches
        for mask_type in MASKS:
            for field in full_fields:
                performance[PERFORMANCE_METRICS[0]][RELEVANT_PARTITIONS[1]][mask_type][field] /= count # loss
                performance[PERFORMANCE_METRICS[1]][RELEVANT_PARTITIONS[1]][mask_type][field] /= (count_token[mask_type] if count_token[mask_type] > 0 else 1) # accuracy

        # output statistics
        logging.info(f"Validation loss: {performance[PERFORMANCE_METRICS[0]][RELEVANT_PARTITIONS[1]][conditional_mask_type][ALL_STRING]:.4f}")
        logging.info("Individual losses: " + ", ".join((f"{field} = {value:.4f}" for field, value in performance[PERFORMANCE_METRICS[0]][RELEVANT_PARTITIONS[1]][conditional_mask_type].items())))

        # log validation info for wandb
        for metric in PERFORMANCE_METRICS:
            for mask_type in MASKS:
                wandb.log({f"{RELEVANT_PARTITIONS[1]}/{metric}/{mask_type}/{field}": value for field, value in performance[metric][RELEVANT_PARTITIONS[1]][mask_type].items()}, step = step)

        ##################################################


        # RECORD LOSS, SAVE MODEL
        ##################################################

        # write output to file
        output = []
        for metric in PERFORMANCE_METRICS:
            for partition in RELEVANT_PARTITIONS:
                for mask_type in MASKS:
                    for field in full_fields:
                        output.append(dict(zip(PERFORMANCE_OUTPUT_COLUMNS, (step, metric, partition, mask_type, field, performance[metric][partition][mask_type][field]))))
        output = pd.DataFrame(data = output, columns = PERFORMANCE_OUTPUT_COLUMNS)
        output.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_VALUE, header = False, index = False, mode = "a")

        # see whether or not to save
        is_an_improvement = False # whether or not the loss has improved
        for partition in RELEVANT_PARTITIONS:
            partition_loss = performance[PERFORMANCE_METRICS[0]][partition][MASKS[1 if args.conditional else 0]][ALL_STRING]
            if partition_loss < min_loss[partition]:
                min_loss[partition] = partition_loss
                logging.info(f"Best {partition}_loss so far!") # log paths to which states were saved
                torch.save(obj = model.state_dict(), f = best_model_filepath[partition]) # save the model
                torch.save(obj = optimizer.state_dict(), f = best_optimizer_filepath[partition]) # save the optimizer state
                torch.save(obj = scheduler.state_dict(), f = best_scheduler_filepath[partition]) # save the scheduler state
                if args.early_stopping: # reset the early stopping counter if we found a better model
                    count_early_stopping = 0
                    is_an_improvement = True # we only care about the lack of improvement when we are thinking about early stopping, so turn off this boolean flag, since there was an improvement
        
        # increment the early stopping counter if no improvement is found
        if (not is_an_improvement) and args.early_stopping:
            count_early_stopping += 1 # increment

        # early stopping
        if args.early_stopping and (count_early_stopping > args.early_stopping_tolerance):
            logging.info(f"Stopped the training for no improvements in {args.early_stopping_tolerance} rounds.")
            break

        ##################################################

    
    # STATISTICS AND CONCLUSION
    ##################################################

    # log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_loss[RELEVANT_PARTITIONS[1]]}")
    wandb.log({f"min_{RELEVANT_PARTITIONS[1]}_loss": min_loss[RELEVANT_PARTITIONS[1]]})

    # finish the wandb run
    wandb.finish()

    # output model name to list of models
    models_output_filepath = f"{dirname(args.output_dir)}/models.txt"
    if exists(models_output_filepath):
        with open(models_output_filepath, "r") as models_output: # read in list of trained models
            models = {model.strip() for model in models_output.readlines()} # use a set because better for `in` operations
    else:
        models = set()
    with open(models_output_filepath, "a") as models_output:
        if basename(args.output_dir) not in models: # check if in list of trained models
            models_output.write(basename(args.output_dir) + "\n") # add model to list of trained models if it isn't already there

    ##################################################

##################################################
