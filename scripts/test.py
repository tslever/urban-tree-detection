'''
Compute metrics on test set and save per-image visualizations.
'''
import numpy as np
import argparse
import os
import h5py as h5
import yaml
from utils.evaluate import evaluate
from models import SFANet
from utils.preprocess import *
import imageio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def save_visualizations(images, results, output_dir):
    '''
    Save one visualization per test image with the test image and overlaid annotations.

    Parameters:
        images: np.ndarray -- Test images array of shape [N, H, W, C]
        results: dict -- Dictionary from function evaluate containing keys `gt_locs`, `tp_locs`, `tp_gt_locs`, `fp_locs`, and `fn_locs`
        output_dir: str -- Path to the directory where visualizations will be saved
    '''
    os.makedirs(output_dir, exist_ok = True)
    num_images = images.shape[0]
    for i in range(0, num_images):
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.imshow(images[i])
        ground_truth_points = results['gt_locs'][i]
        if ground_truth_points.size > 0:
            if len(ground_truth_points.shape) == 1:
                ground_truth_points = ground_truth_points[None, :]
            ax.plot(ground_truth_points[:,0], ground_truth_points[:,1], 'm.', label = 'Ground Truth')

        true_positives = results['tp_locs'][i]
        if true_positives.size > 0:
            if len(true_positives.shape) == 1:
                true_positives = true_positives[None, :]
            ax.plot(true_positives[:,0], true_positives[:,1], 'g+', label = 'True Positives')

        false_positives = results['fp_locs'][i]
        if false_positives.size > 0:
            if len(false_positives.shape) == 1:
                false_positives = false_positives[None, :]
            ax.plot(false_positives[:,0], false_positives[:,1], 'y^', label = 'False Positives')

        false_negatives = results['fn_locs'][i]
        if false_negatives.size > 0:
            if len(false_negatives.shape) == 1:
                false_negatives = false_negatives[None, :]
            ax.plot(false_negatives[:,0], false_negatives[:,1], 'm.', markeredgecolor='k', markeredgewidth=1, label='False Negatives')

        true_positives_and_ground_truths = results['tp_gt_locs'][i]
        if true_positives.size > 0 and true_positives_and_ground_truths.size > 0:
            if len(true_positives.shape) == 1:
                true_positives_and_ground_truths = true_positives_and_ground_truths[None, :]
            for (x1, y1), (x2, y2) in zip(true_positives, true_positives_and_ground_truths):
                ax.plot([x1, x2], [y1, y2], 'y-')

        ax.axis('off')
        save_path = os.path.join(output_dir, f'image_{i:04d}.png')
        fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--max_distance', type=float, default=10, help='max distance from ground truth to pred tree (in pixels)')

    args = parser.parse_args()

    params_path = os.path.join(args.log, 'params.yaml')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            mode = params['mode']
            min_distance = params['min_distance']
            threshold_abs = params['threshold_abs'] if mode == 'abs' else None
            threshold_rel = params['threshold_rel'] if mode == 'rel' else None
    else:
        print(f'warning: params.yaml missing -- using default params')
        min_distance = 1
        threshold_abs = None
        threshold_rel = 0.2
    
    f = h5.File(args.data, 'r')
    images = f[f'test/images'][:]
    gts = f[f'test/gt'][:]

    bands = f.attrs['bands']
    
    preprocess = eval(f'preprocess_{bands}')
    training_model, model = SFANet.build_model(
        images.shape[1:],
        preprocess_fn = preprocess
    )

    weights_path = os.path.join(args.log, 'best.weights.h5')
    training_model.load_weights(weights_path)

    print('----- getting predictions from trained model -----')
    preds = model.predict(images, verbose = True, batch_size = 1)[..., 0]

    print('----- calculating metrics -----')
    results = evaluate(
        gts=gts,
        preds=preds,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        max_distance=args.max_distance,
        return_locs=True)

    with open(os.path.join(args.log,'results.txt'),'w') as f:
        f.write('precision: '+str(results['precision'])+'\n')
        f.write('recall: '+str(results['recall'])+'\n')
        f.write('fscore: '+str(results['fscore'])+'\n')
        f.write('rmse [px]: '+str(results['rmse'])+'\n')

    print('------- results for: ' + args.log + ' ---------')
    print('precision: ',results['precision'])
    print('recall: ',results['recall'])
    print('fscore: ',results['fscore'])
    print('rmse [px]: ',results['rmse'])
    
    vis_dir = os.path.join(args.log, 'visualizations')
    save_visualizations(images, results, vis_dir)
    print('Visualizations saved to directory:', vis_dir)


if __name__ == '__main__':
    main()
