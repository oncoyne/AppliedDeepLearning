#!/usr/bin/env python3
import torch
import argparse
import torch.backends.cudnn
import torchvision.datasets

from utils import *
from dataset import Salicon
from model import CNN
from pathlib import Path
from train import Trainer
from torch import nn, optim
from torchvision import transforms
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a CNN on SALICON",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Parser arguments
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.05, type=float, help="Learning rate")
parser.add_argument("--batch-size", default=128, type=int, help="Number of images within each mini-batch")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs (passes through the entire dataset) to train for")
parser.add_argument("--val-frequency", default=2, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--acc-frequency", 
                    default=200, 
                    type=int, 
                    help="How frequently to calcualte the accuracy on the validation set and output predictions in number of epochs"
                    )
parser.add_argument("--log-frequency", default=5, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-frequency", default=157, type=int, help="How frequently to print progress to the command line in number of steps")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")
parser.add_argument("--checkpoint-path", default=Path("checkpoints"), type=Path, help="Checkpoint path")
parser.add_argument("--checkpoint-frequency", type=int, default=50, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path, help="Resume from given check point")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    # Load the data
    train_loader = torch.utils.data.DataLoader(
        Salicon("train.pkl"),
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        Salicon("val.pkl"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True
    )

    # Initialise network model
    model = CNN(height=96, width=96, channels=3)

    start = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint)
        start=checkpoint['current_epoch']
        print("Resuming model", args.resume_checkpoint, "from epoch", start, "which had loss", checkpoint['loss'])
        model.load_state_dict(checkpoint['model'])
        args = checkpoint['args']
        print("Model:", model)
        print("Arguments are:", args)
        

    # Initialise loss function - Mean Squared Error
    criterion = nn.MSELoss()

    # Initialise optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)


    # Initialise learning rates and momentums
    lrs = np.linspace(args.learning_rate, 0.0001, args.epochs)
    moms = np.linspace(0.9, 0.999, args.epochs)

    # Initialise log writer
    log_dir = get_summary_writer_log_dir(args)
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    # Initialise checkpoint directory
    checkpoint_dir = initialise_checkpoint_directory(args)

    # Initialise preds directiory
    preds_dir = initialise_preds_directory()

    print("Writing logs to", log_dir)
    print("Writing checkpoints to", checkpoint_dir)
    print("Writing preds to", preds_dir)

    # Initialise trainer and train on data 
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE, preds_dir)
    trainer.train(args, lrs, moms, start_epoch=start, checkpoint_directory=checkpoint_dir)

    summary_writer.close()


if __name__ == "__main__":
    main(parser.parse_args())