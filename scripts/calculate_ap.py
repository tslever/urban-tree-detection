""" Compute average precision on test set. """
import numpy as np
import argparse
import os
import h5py as h5
import yaml
from utils.evaluate import test_all_thresholds, calculate_ap
from models import SFANet, SFANetRes, SFANetEfficient
from utils.preprocess import *
import imageio
import json
with open('config.json', 'r') as f:
    config = json.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--max_distance', type=float, default=10, help='max distance from gt to pred tree (in pixels)')
    parser.add_argument('--center_crop', action = 'store_true', help = 'Evaluate only on the center 166 x 166 pixels of each test image')
    
    args = parser.parse_args()

    f = h5.File(args.data, 'r')
    images = f[f'test/images'][:]
    gts = f[f'test/gt'][:]
    
    preds_path = os.path.join(args.log, 'test_preds.npy')
    if os.path.exists(preds_path):
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

        print('----- getting predictions from trained model -----')
        preds = model.predict(images, verbose = True, batch_size = 1)[..., 0]
        
        np.save(preds_path, preds)

    if args.center_crop:
        crop_size = 166
        def center_crop(arr):
            H, W = arr.shape[0], arr.shape[1]
            start_H = (H - crop_size) // 2
            start_W = (W - crop_size) // 2
            return arr[start_H:start_H+crop_size, start_W:start_W+crop_size, ...]
        gts = np.array([center_crop(gt) for gt in gts])
        preds = np.array([center_crop(pred) for pred in preds])

    print('----- calculating metrics -----')
    thresholds, precisions, recalls = test_all_thresholds(
        gts = gts,
        preds = preds,
        max_distance = args.max_distance
    )
    ap = calculate_ap(precisions, recalls)

    with open(os.path.join(args.log, 'ap_results.txt'), 'w') as f:
        f.write('average precision: ' + str(ap))

    print('------- results for: ' + args.log + ' ---------')
    print('average precision: ', ap)

if __name__ == '__main__':
    main()
