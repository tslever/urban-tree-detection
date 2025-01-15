import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

import glob
import numpy as np

from models import SFANet
from utils.preprocess import *

import argparse
import os
import sys

import h5py as h5

class DataGenerator(Sequence):
    def __init__(self, f, batch_size, shuffle=True, **kwargs):
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
    parser.add_argument('--epochs', type=int, default=500, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args = parser.parse_args()

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

    f = h5.File(args.data,'r')
    bands = f.attrs['bands']
    val_images = f['val/images'][:]
    val_confidence = f['val/confidence'][:]
    val_attention = f['val/attention'][:]
    
    preprocess_fn = eval(f'preprocess_{bands}')
    
    model, testing_model = SFANet.build_model(
        val_images.shape[1:],
        preprocess_fn=preprocess_fn)
    opt = Adam(args.lr)
    model.compile(optimizer=opt, loss=['mse','binary_crossentropy'], loss_weights=[1,0.1])

    print(model.summary())
    
    os.makedirs(args.log,exist_ok=True)

    callbacks = []

    weights_path = os.path.join(args.log, 'best.weights.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=True,
            save_weights_only=True,
            ))
    weights_path = os.path.join(args.log, 'latest.weights.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=False,
            save_weights_only=True,
            ))
    tensorboard_path = os.path.join(args.log,'tensorboard')
    os.system("rm -rf " + tensorboard_path)
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))

    gen = DataGenerator(f,args.batch_size,shuffle=True,use_multiprocessing=True,workers=4)
    y_val = (val_confidence, val_attention)

    model.fit(
            gen,
            validation_data=(val_images,y_val),
            epochs=args.epochs,
            verbose=True,
            callbacks=callbacks
    )

if __name__ == '__main__':
    main()
