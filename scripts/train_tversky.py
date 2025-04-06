import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

import argparse
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from models import SFANet
from utils.preprocess import *

#new meant to speed up
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
#

def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
    false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
    false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)

    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1.0 - tversky_index

class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, file_path, monitor='val_loss', patience=20, mode='min'):
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

    def on_epoch_end(self, epoch, logs=None):
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
                    f.write(f"Early stopping triggered at epoch {epoch + 1}.\n")


class LossHistoryCSV(tf.keras.callbacks.Callback):
    def __init__(self, csv_path, resume=False):
        super().__init__()
        self.csv_path = csv_path
        if not resume or not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("epoch,train_loss,val_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        with open(self.csv_path, 'a') as f:
            f.write(f"{epoch + 1},{train_loss},{val_loss}\n")
        print(f"Epoch {epoch + 1} loss written to {self.csv_path}")


class LossCurvePlotter(tf.keras.callbacks.Callback):
    def __init__(self, png_path):
        super().__init__()
        self.png_path = png_path
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))

        plt.figure()
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.savefig(self.png_path)
        plt.close()


class DataGenerator(Sequence):
    def __init__(self, f, batch_size, shuffle=True):
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

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = np.stack([self.train_images[i] for i in batch_idx])
        batch_confidence = np.stack([self.train_confidence[i] for i in batch_idx])
        batch_attention = np.stack([self.train_attention[i] for i in batch_idx])
        return batch_images, (batch_confidence, batch_attention)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    tf.config.optimizer.set_jit(True) #new
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to training HDF5 file")
    parser.add_argument("log", help="Path to logging directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    #tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")

    f = h5.File(args.data, "r")
    bands = f.attrs["bands"]

    if "val/images" in f:
        val_images = f["val/images"][:]
        val_confidence = f["val/confidence"][:]
        val_attention = f["val/attention"][:]
    else:
        print("No validation set found. Using training set as validation.")
        val_images = f["train/images"][:]
        val_confidence = f["train/confidence"][:]
        val_attention = f["train/attention"][:]

    preprocess_fn = eval(f"preprocess_{bands}")
    model, testing_model = SFANet.build_model(val_images.shape[1:], preprocess_fn=preprocess_fn)

    if model is None:
        raise ValueError("SFANet.build_model returned None — check implementation!")

    model.compile(optimizer=Adam(args.lr), loss=[tversky_loss, "binary_crossentropy"], loss_weights=[1, 0.1])
    print(model.summary())

    os.makedirs(args.log, exist_ok=True)

    best_weights_path = os.path.join(args.log, "best.weights.h5")
    latest_weights_path = os.path.join(args.log, "latest.weights.h5")
    tensorboard_path = os.path.join(args.log, "tensorboard")
    csv_file = os.path.join(args.log, "loss_history.csv")
    loss_curves_path = os.path.join(args.log, "loss_curves.png")
    early_stopping_log_path = os.path.join(args.log, "early_stopping_log.txt")

    callbacks = [
        ModelCheckpoint(best_weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
        ModelCheckpoint(latest_weights_path, monitor="val_loss", save_best_only=False, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(tensorboard_path),
        LossHistoryCSV(csv_file, resume=args.resume),
        LossCurvePlotter(loss_curves_path),
        EarlyStopping(early_stopping_log_path, monitor="val_loss", patience=20)
    ]

    initial_epoch = 0
    if args.resume:
        if os.path.exists(latest_weights_path):
            if os.path.exists(early_stopping_log_path):
                with open(early_stopping_log_path) as f_log:
                    if "Early stopping triggered" in f_log.read():
                        raise Exception("Early stopping occurred. Restart training from scratch.")
            if not os.path.exists(csv_file):
                raise Exception("Missing loss history CSV file.")
            with open(csv_file) as f_csv:
                lines = f_csv.readlines()
                if len(lines) <= 1:
                    raise Exception("CSV file does not contain epoch history.")
                initial_epoch = int(lines[-1].split(",")[0])
            print(f"Resuming from epoch {initial_epoch}")
            model.load_weights(latest_weights_path)
        else:
            raise Exception("No checkpoint found to resume from.")

    gen = DataGenerator(f, args.batch_size, shuffle=True)
    y_val = (val_confidence, val_attention)

    model.fit(
        gen,
        validation_data=(val_images, y_val),
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        verbose=True,
        callbacks=callbacks,
        #use_multiprocessing=False,  # safer for HDF5
        #max_queue_size=10  # Optional: controls the generator queue size
    )


if __name__ == "__main__":
    main()