import os
import time
import torch
import model
import pickle
import evaluation
import numpy as np

from utils import *
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        preds_dir: str
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.preds_dir = preds_dir

    def train(self, args, lrs, moms, start_epoch: int = 0, checkpoint_directory: str = ""):
        for epoch in range(start_epoch, args.epochs):
            self.model.train()
            data_load_start_time = time.time()
            for group in self.optimizer.param_groups:
                group['lr'] = lrs[epoch] # Adaptive Learning rate
                group['momentum'] = moms[epoch] # Momentum Decay
                
            for batch, labels in self.train_loader:
                # Load Batch
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                data_load_end_time = time.time()
            
                # Forward pass on the input
                logits = self.model.forward(batch)
                # Measure loss and backprop
                loss = self.criterion(logits, labels)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % args.log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % args.print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            
            # Check if validation, accuracy, saving predictions or checkpoints
            validate_model = (((epoch + 1) % args.val_frequency) == 0) or (epoch == 0) or (epoch + 1 == args.epochs)
            calc_accuracy = (((epoch + 1) % args.acc_frequency) == 0) or (epoch == 0) or (epoch + 1 == args.epochs)
            save_checkpoint = ((epoch + 1) % args.checkpoint_frequency == 0) or ((epoch + 1) == args.epochs) 
            
            if save_checkpoint:
                self.checkpoint_model(checkpoint_directory=checkpoint_directory, epoch=epoch, loss=loss, args=args, lrs=lrs)
            if (validate_model or calc_accuracy):
                self.validate(validate_model, calc_accuracy)
                self.model.train() # Switch back to train mode
                

    def checkpoint_model(self, checkpoint_directory: str, epoch: int, loss, args: argparse.Namespace, lrs: tuple):
        checkpoint_path = checkpoint_directory + "/epoch_checkpoint_" + str(epoch)
        print(f"Saving model to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)
        args.learning_rate = lrs[epoch]
        torch.save({'args': args, 'model': self.model.state_dict(), 'loss': loss.values, 'current_epoch': epoch}, checkpoint_path)

    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step + 1}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars("loss", {"train": float(loss.item())}, self.step)
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def validate(self, validate_model, calc_accuracy):
        self.model.eval()
        if validate_model:
            preds = []
            total_loss = 0
            # No need to track gradients for validation, we're not optimizing.
            with torch.no_grad():
                for batch, labels in self.val_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(batch)

                    loss = self.criterion(logits, labels)
                    preds.extend(logits.cpu().numpy())

                    total_loss += loss.item()
            average_loss = total_loss / len(self.val_loader)       
            print(f"Validation loss: {average_loss:.5f}")
            self.summary_writer.add_scalars("loss", {"test": average_loss}, self.step)
        
        # Calculate accuracy and output predictions
        if calc_accuracy:
            preds_path = get_preds_path(self.preds_dir)
            with open(preds_path,'wb') as f:
                    pickle.dump(preds, f)
            cc, auc_borj, auc_shuffle = evaluation.evaluate(preds_path, "val.pkl")
            self.summary_writer.add_scalars("accuracy", {"CC": cc, "AUC Borj": auc_borj, "AUC SHUFFLED": auc_shuffle}, self.step)
        

