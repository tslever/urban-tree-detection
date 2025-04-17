""" Run hyperparameter tuning on validation set to determine optimal detection parameters. """

from utils.evaluate import evaluate
import argparse
import os
import h5py as h5
from models import SFANet, SFANetRes, SFANetEfficient
from utils.preprocess import *
import optuna
import yaml
import numpy as np

import json
with open('config.json', 'r') as f:
    config = json.load(f)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--ntrials', type=int, default=200, help='number of trials')
    parser.add_argument('--max_distance', type=float, default=10, help='max distance from gt to pred tree (in pixels)')

    args = parser.parse_args()

    f = h5.File(args.data,'r')
    images = f['val/images'][:]
    gts = f['val/gt'][:]

    preds_path = os.path.join(args.log,'val_preds.npy')
    if os.path.exists(preds_path):
        print('----- loading predictions from file -----')
        preds = np.load(preds_path)
    else:
        bands = f.attrs['bands']
        preprocess = eval(f'preprocess_{bands}')

        model_type = config.get("model", "vgg")

        if model_type == 'vgg':
            model_builder = SFANet
        elif model_type == 'resnet':
            model_builder = SFANetRes
        elif model_type == 'efficientnet':
            model_builder = SFANetEfficient
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        training_model, model = model_builder.build_model(images.shape[1:], preprocess_fn=preprocess)

        weights_path = os.path.join(args.log, 'best.weights.h5')
        training_model.load_weights(weights_path)



        weights_path = os.path.join(args.log,'best.weights.h5')
        training_model.load_weights(weights_path)

        print('----- getting predictions from trained model -----')
        preds_list = []
        for i in range(len(images)):
            pred = model.predict(np.expand_dims(images[i], axis = 0), verbose = False)
            preds_list.append(pred[0, ..., 0])
        preds = np.array(preds_list)
        
        np.save(preds_path, preds)

    def objective(trial):
        min_distance = trial.suggest_int('min_distance',1,10)
        mode = trial.suggest_categorical('mode',['abs','rel'])
        threshold_abs = trial.suggest_float('threshold_abs',-10,10)
        threshold_rel = trial.suggest_float('threshold_rel',0,1)
        results = evaluate(
            gts=gts,
            preds=preds,
            min_distance=min_distance,
            threshold_rel=threshold_rel if mode=='rel' else None,
            threshold_abs=threshold_abs if mode=='abs' else None,
            max_distance=args.max_distance)
        return 1 - results['fscore']

    print('----- running hyperparameter tuning -----')
    study = optuna.create_study()
    study.optimize(objective, n_trials = args.ntrials)

    print('----- best params: -----')
    print(study.best_params)

    output_path = os.path.join(args.log, 'params.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(study.best_params, f)

if __name__ == '__main__':
    main()
