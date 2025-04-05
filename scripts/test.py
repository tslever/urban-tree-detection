'''
Compute metrics on test set and save per-image visualizations and
a scatter plot of predicted vs. actual dead trees per hectare.
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
import re
import rasterio
import json


def get_source_image_path(tile_path):
    '''
    Given a testing tile path, return the corresponding source image.
    For example, if tile_path is:
        ../urban-tree-detection-data/images/testing_image_0_CT_1_bl_reproj_m_4107348_ne_18_060_20180810_0.tif
    then the source image path is:
        ../urban-tree-detection-data/stacked_testing_images/testing_image_0_CT_1_bl_reproj_m_4107348_ne_18_060_20180810.tif
    '''
    dirname, basename = os.path.split(tile_path)
    base_no_ext, ext = os.path.splitext(basename)
    source_base = re.sub(r'_\d+$', '', base_no_ext) + ext
    source_dir = dirname.replace('images', 'stacked_testing_images')
    source_path = os.path.join(source_dir, source_base)
    return source_path


def save_visualizations(images, results, names, output_dir, rearrange_channels = False, dpi = 100):
    '''
    Save one visualization per test image with the test image and overlaid annotations.
    Ensure that the saved visualization has the same resolution as the input image.

    Parameters:
        images: np.ndarray -- Test images array of shape [N, H, W, C]
        results: dict -- Dictionary from function evaluate containing keys `gt_locs`, `tp_locs`, `tp_gt_locs`, `fp_locs`, and `fn_locs`
        names: list -- List of names corresponding to each test image
        output_dir: str -- Path to the directory where visualizations will be saved
        rearrange_channels: bool -- If True, rearrange channels (for example, use channel order [3, 0, 1] if available)
        dpi: int -- DPI to use for the figure (affects figure size in inches)
    '''
    os.makedirs(output_dir, exist_ok = True)
    num_images = len(images)
    for i in range(0, num_images):
        img = images[i]
        if rearrange_channels:
            if img.shape[-1] >= 4:
                img = img[..., [3, 0, 1]]
            else:
                img = img[..., :3]
        height, width = img.shape[:2]
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)

        ground_truth_points = results['gt_locs'][i]
        if ground_truth_points.size > 0:
            if ground_truth_points.ndim == 1:
                ground_truth_points = ground_truth_points[None, :]
            ax.plot(ground_truth_points[:, 0], ground_truth_points[:, 1], 'm.', label = 'Ground Truth')

        true_positives = results['tp_locs'][i]
        if true_positives.size > 0:
            if true_positives.ndim == 1:
                true_positives = true_positives[None, :]
            ax.plot(true_positives[:, 0], true_positives[:, 1], 'g+', label = 'True Positives')

        false_positives = results['fp_locs'][i]
        if false_positives.size > 0:
            if false_positives.ndim == 1:
                false_positives = false_positives[None, :]
            ax.plot(false_positives[:, 0], false_positives[:, 1], 'y^', label = 'False Positives')

        false_negatives = results['fn_locs'][i]
        if false_negatives.size > 0:
            if false_negatives.ndim == 1:
                false_negatives = false_negatives[None, :]
            ax.plot(false_negatives[:, 0], false_negatives[:, 1], 'm.', markeredgecolor = 'k', markeredgewidth = 1, label = 'False Negatives')

        true_positives_and_ground_truths = results['tp_gt_locs'][i]
        if true_positives.size > 0 and true_positives_and_ground_truths.size > 0:
            if true_positives_and_ground_truths.ndim == 1:
                true_positives_and_ground_truths = true_positives_and_ground_truths[None, :]
            for (x1, y1), (x2, y2) in zip(true_positives, true_positives_and_ground_truths):
                ax.plot([x1, x2], [y1, y2], 'y-')

        ax.axis('off')
        save_path = os.path.join(output_dir, f'{names[i]}.png')
        fig.savefig(save_path, dpi = dpi, pad_inches = 0)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help = 'path to data hdf5 file')
    parser.add_argument('log', help = 'path to log directory')
    parser.add_argument('--max_distance', type = float, default = 10, help = 'max distance from ground truth to pred tree (in pixels)')
    parser.add_argument('--rearrange_channels', action = 'store_true', help = 'Rearrange image channels for visualization if provided')
    parser.add_argument('--center_crop', action = 'store_true', help = 'Evaluate only on the center 166 x 166 pixels of each test image.')

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

    if 'names' in f['test']:
        raw_names = f['test/names'][:]
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in raw_names]
    else:
        names = [f'image_{i:04d}' for i in range(images.shape[0])]

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

    if args.center_crop:
        crop_size = 166
        def center_crop(arr, metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            sub_height = metadata['sub_height']
            sub_width = metadata['sub_width']
            start_H = (sub_height - crop_size) // 2
            start_W = (sub_width - crop_size) // 2
            return arr[start_H:start_H+crop_size, start_W:start_W+crop_size, ...]
        
        cropped_images = []
        cropped_gts = []
        cropped_preds = []
        for i, (img, gt, pred) in enumerate(zip(images, gts, preds)):
            metadata_path = os.path.join("../urban-tree-detection-data/images_based_on_chopped_testing_images", f"{names[i]}.json")
            cropped_images.append(center_crop(img, metadata_path))
            cropped_gts.append(center_crop(gt, metadata_path))
            cropped_preds.append(center_crop(pred, metadata_path))
        images = np.array(cropped_images)
        gts = np.array(cropped_gts)
        preds = np.array(cropped_preds)

    print('----- calculating metrics -----')
    results = evaluate(
        gts = gts,
        preds = preds,
        min_distance = min_distance,
        threshold_rel = threshold_rel,
        threshold_abs = threshold_abs,
        max_distance = args.max_distance,
        return_locs = True
    )

    with open(os.path.join(args.log, 'results.txt'), 'w') as f:
        f.write('precision: ' + str(results['precision']) + '\n')
        f.write('recall: ' + str(results['recall']) + '\n')
        f.write('fscore: ' + str(results['fscore']) + '\n')
        f.write('rmse [px]: ' + str(results['rmse']) + '\n')

    print('------- results for: ' + args.log + ' ---------')
    print('precision: ', results['precision'])
    print('recall: ', results['recall'])
    print('fscore: ', results['fscore'])
    print('rmse [px]: ', results['rmse'])
    
    '''
    Compute number of actual dead trees and number of predicted dead trees.
    Determine numbers of actual and predicted dead trees per hectare.
    The number of predicted dead trees is the sum of the number of true positives and the number of false positives.
    '''
    numbers_of_actual_dead_trees_per_hectare = []
    numbers_of_predicted_dead_trees_per_hectare = []
    for gt_locs, tp_locs, fp_locs in zip(results['gt_locs'], results['tp_locs'], results['fp_locs']):
        number_of_actual_dead_trees = len(gt_locs)
        number_of_predicted_dead_trees = len(tp_locs) + len(fp_locs)
        '''
        Calculate the number of hectares in the testing image as
        (height in pixels) * (0.6 meters / pixel) * (width in pixels) * (0.6 meters / pixel) * (1 hectare / (10000 square meters)).
        '''
        height, width = images[i].shape[:2]
        area_in_hectares = (height * 0.6 * width * 0.6) / 10000.0
        numbers_of_actual_dead_trees_per_hectare.append(number_of_actual_dead_trees / area_in_hectares)
        numbers_of_predicted_dead_trees_per_hectare.append(number_of_predicted_dead_trees / area_in_hectares)
    
    plt.figure()
    plt.scatter(
        numbers_of_actual_dead_trees_per_hectare,
        numbers_of_predicted_dead_trees_per_hectare,
        c = 'blue',
        edgecolors = 'k',
        alpha = 0.2
    )
    plt.xlabel("number of actual dead trees per hectare")
    plt.ylabel("number of predicted dead trees per hectare")
    plt.title(
        "Number of Predicted Dead Trees Per Hectare\n" +
        "vs. Number of Actual Dead Trees Per Hectare"
    )
    all_numbers_of_dead_trees_per_hectare = numbers_of_actual_dead_trees_per_hectare + numbers_of_predicted_dead_trees_per_hectare
    minimum_number_of_dead_trees_per_hectare = min(all_numbers_of_dead_trees_per_hectare)
    maximum_number_of_dead_trees_per_hectare = max(all_numbers_of_dead_trees_per_hectare)
    plt.plot(
        [minimum_number_of_dead_trees_per_hectare, maximum_number_of_dead_trees_per_hectare],
        [minimum_number_of_dead_trees_per_hectare, maximum_number_of_dead_trees_per_hectare],
        'r--',
        label = 'y = x'
    )
    plt.legend()
    plt.grid()
    path_to_scatter_plot = os.path.join(args.log, "scatter_plot.png")
    plt.savefig(path_to_scatter_plot)
    plt.close()
    print("Scatter plot was saved to " + path_to_scatter_plot)
    
    vis_dir = os.path.join(args.log, 'visualizations')
    save_visualizations(images, results, names, vis_dir, rearrange_channels = args.rearrange_channels)
    print('Visualizations saved to directory:', vis_dir)


if __name__ == '__main__':
    main()
