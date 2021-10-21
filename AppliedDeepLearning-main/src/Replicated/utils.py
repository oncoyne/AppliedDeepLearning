import os
import torch
import argparse
import numpy as np

from torch import nn
from typing import Union
from torch.nn import functional as F

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = "Shallow_CNN_Salicon_run_"
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def initialise_checkpoint_directory(args: argparse.Namespace) -> str:
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    checkpoint_dir_prefix = str(args.checkpoint_path) + "/Shallow_CNN_Salicon_run_"
    checkpoint_dir = ""
    i = 0
    while i < 1000:
        checkpoint_dir = checkpoint_dir_prefix + str(i)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            break
        i += 1
    return checkpoint_dir

def initialise_preds_directory() -> str:
    if not os.path.exists("preds"):
        os.mkdir("preds")

    pred_dir_prefix = "preds/Shallow_CNN_Salicon_run_"
    pred_dir = ""
    i = 0
    while i < 1000:
        pred_dir = pred_dir_prefix + str(i)
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
            break
        i += 1
    return pred_dir

def get_preds_path(preds_dir: str) -> str:
    preds_dir_prefix = preds_dir + "/validation_round_" 
    preds_path = ""
    i = 0
    while i < 1000:
        preds_path = preds_dir_prefix + str(i)
        if not os.path.exists(preds_path):
            os.mkdir(preds_path)
            break
        i += 1

    print(f"Validation round {i}: Saving to {preds_path}")
    return (preds_path + "/preds.pkl")