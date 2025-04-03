import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

import glob
import numpy as np

from models import SFANetRes
from utils.preprocess import *

import argparse
import os
import sys

import h5py as h5

import matplotlib.pyplot as plt


class EarlyStopping(tf.keras.callbacks.Callback):
    
    def __init__(self, file_path, monitor = 'val_loss', patience = 20, mode = 'min'):
        super().__init__()
        self.file_path = file_path
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        with open(self.file_path, 'w') as f:
            f.write("Early stopping has not occurred.\n")

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                with open(self.file_path, 'w') as f:
                    f.write(f"Early stopping triggered at epoch {epoch+1}.\n")


class LossHistoryCSV(tf.keras.callbacks.Callback):

    def __init__(self, csv_path, resume = False):
        super().__init__()
        self.csv_path = csv_path
        if not resume or not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline = "") as f:
                f.write("epoch,train_loss,val_loss\n")
    
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        with open(self.csv_path, 'a', newline = "") as f:
            f.write(f"{epoch+1},{train_loss},{val_loss}\n")
        print(f"Epoch {epoch+1} loss was logged to {self.csv_path}")

        
class LossCurvePlotter(tf.keras.callbacks.Callback):

    def __init__(self, png_path):
        super().__init__()
        self.png_path = png_path
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        plt.figure()
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label = "Training Loss")
        plt.plot(epochs, self.val_losses, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Curves")
        plt.savefig(self.png_path)
        plt.close()
        print(f"Epoch {epoch+1} loss curves updated to {self.png_path}")

class DataGenerator(Sequence):

    def __init__(self, f, batch_size, shuffle = True, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_images = f['train/images']
        self.train_confidence = f['train/confidence']
        self.train_attention = f['train/attention']
        self.indices = np.arange(len(self.train_images))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.train_images) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = np.stack([self.train_images[i] for i in batch_indices])
        batch_confidence = np.stack([self.train_confidence[i] for i in batch_indices])
        batch_attention = np.stack([self.train_attention[i] for i in batch_indices])
        return batch_images, (batch_confidence, batch_attention)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to training data hdf5 file')
    parser.add_argument('log', help='path to log directory')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default = 1_000_000, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint if early stopping has not occurred')

    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(e)

    f = h5.File(args.data, 'r')
    bands = f.attrs['bands']

    if 'val/images' in f:
        val_images = f['val/images'][:]
        val_confidence = f['val/confidence'][:]
        val_attention = f['val/attention'][:]
    else:
        print("Warning: Validation data not found in HDF5 file. Using training data as validation.")
        val_images = f['train/images'][:]
        val_confidence = f['train/confidence'][:]
        val_attention = f['train/attention'][:]
    
    preprocess_fn = eval(f'preprocess_{bands}')
    
    model, testing_model = SFANetRes.build_model(
        val_images.shape[1:],
        preprocess_fn=preprocess_fn
    )
    opt = Adam(args.lr)
    model.compile(optimizer = opt, loss = ['mse', 'binary_crossentropy'], loss_weights = [1, 0.1])

    print(model.summary())
    
    os.makedirs(args.log, exist_ok = True)

    callbacks = []

    best_weights_path = os.path.join(args.log, 'best.weights.h5')
    callbacks.append(
        ModelCheckpoint(
            filepath = best_weights_path,
            monitor = 'val_loss',
            verbose = True,
            save_best_only = True,
            save_weights_only = True,
        )
    )
    latest_weights_path = os.path.join(args.log, 'latest.weights.h5')
    callbacks.append(
        ModelCheckpoint(
            filepath = latest_weights_path,
            monitor = 'val_loss',
            verbose = True,
            save_best_only = False,
            save_weights_only = True
        )
    )
    tensorboard_path = os.path.join(args.log, 'tensorboard')
    os.system("rm -rf " + tensorboard_path)
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))

    csv_file = os.path.join(args.log, "loss_history.csv")
    callbacks.append(LossHistoryCSV(csv_file, resume = args.resume))
    
    loss_curves_path = os.path.join(args.log, "loss_curves.png")
    callbacks.append(LossCurvePlotter(loss_curves_path))
    
    early_stopping_log_path = os.path.join(args.log, 'early_stopping_log.txt')
    early_stopping_callback = EarlyStopping(early_stopping_log_path, monitor = 'val_loss', patience = 20, mode = 'min')
    callbacks.append(early_stopping_callback)

    initial_epoch = 0
    if args.resume:
        if os.path.exists(latest_weights_path):
            if os.path.exists(early_stopping_log_path):
                with open(early_stopping_log_path, 'r') as f_log:
                    content = f_log.read()
                if "Early stopping triggered" in content:
                    raise Exception("Early stopping has occurred previously. Please restart training from scratch.")
            if not os.path.exists(csv_file):
                raise Exception("Loss history CSV file does not exist. Please restart training from scratch.")
            with open(csv_file, 'r') as f_csv:
                lines = f_csv.readlines()
                print(lines)
                if len(lines) <= 1:
                    raise Exception("Loss history CSV file does not contain epoch information. Please restart training from scratch.")
                last_line = lines[-1]
                try:
                    initial_epoch = int(last_line.split(',')[0])
                except Exception:
                    raise Exception("Could not parse the last epoch from CSV. Please restart training from scratch.")
            print("Resuming training from latest checkpoint.")
            model.load_weights(latest_weights_path)
            print(f"Resuming from epoch {initial_epoch}.")
        else:
            raise Exception("No latest checkpoint found. Please restart training from scratch.")

    gen = DataGenerator(f, args.batch_size, shuffle = True, use_multiprocessing = True, workers = 4)
    y_val = (val_confidence, val_attention)

    model.fit(
        gen,
        validation_data = (val_images, y_val),
        epochs = args.epochs,
        initial_epoch = initial_epoch,
        verbose = True,
        callbacks = callbacks
    )
    

if __name__ == '__main__':
    main()
