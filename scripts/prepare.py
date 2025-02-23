import argparse
import os
import imageio
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import tqdm

def process_image(dataset_path, name, sigma, bands):
    image = None
    for suffix in ['.tif', '.tiff', '.png']:
        image_path = os.path.join(dataset_path, 'images', name + suffix)
        if os.path.exists(image_path):
            image = imageio.imread(image_path)
            if image.ndim == 3 and image.shape[0] in [3, 4]:
                image = image.transpose(1, 2, 0)
            if suffix == '.png' or bands == 'RGB':
                image = image[..., :3]
            break
    if image is None:
        raise RuntimeError(f'Could not find image for {name}')

    csv_path = os.path.join(dataset_path, 'csv', name + '.csv')
    if os.path.exists(csv_path):
        points = np.loadtxt(csv_path, delimiter = ',', skiprows = 1).astype('int')
        if points.size == 0:
            print(f"Warning: No points found in {csv_path}. Using empty ground truth.")
            gt = np.zeros(image.shape[:2], dtype = 'float32')
        else:
            if points.ndim == 1:
                points = points[None, :]
            gt = np.zeros(image.shape[:2], dtype = 'float32')
            gt[points[:, 1], points[:, 0]] = 1
        distance = distance_transform_edt(1 - gt).astype('float32')
        confidence = np.exp(-distance**2 / (2 * sigma**2))
    else:
        gt = np.zeros(image.shape[:2], dtype = 'float32')
        confidence = np.zeros(image.shape[:2], dtype = 'float32')
    confidence = confidence[..., None]
    attention = (confidence > 0.001).astype('float32').squeeze(axis = -1)
    return image, gt, confidence, attention

def augment_images(images):
    '''
    Return an augmented stack of 8 versions of the image, including 4 90 degree rotations and the vertical flip of each.
    '''
    augmented = []
    for k in range(0, 4):
        rotated = np.rot90(image, k = k)
        augmented.append(rotated)
        augmented.append(np.flipud(rotated))
    return np.stack(augmented)

def process_split(f, dataset_path, split_file, split, sigma, bands, augment = False):
    with open(os.path.join(dataset_path, split_file), 'r') as sf:
        names = [line.strip() for line in sf]
    sample_img, sample_gt, sample_conf, sample_att = process_image(dataset_path, names[0], sigma, bands)
    H, W = sample_img.shape[:2]
    C = sample_img.shape[2]
    num_images = len(names)
    if augment:
        total_images = num_images * 8
    else:
        total_images = num_images
    dset_img = f.create_dataset(f"{split}/images", shape = (0, H, W, C), maxshape = (total_images, H, W, C), dtype = sample_img.dtype, chunks = True)
    dset_gt = f.create_dataset(f"{split}/gt", shape = (0, H, W), maxshape = (total_images, H, W), dtype = 'float32', chunks = True)
    dset_conf = f.create_dataset(f"{split}/confidence", shape = (0, H, W, 1), maxshape = (total_images, H, W, 1), dtype = 'float32', chunks = True)
    dset_att = f.create_dataset(f"{split}/attention", shape = (0, H, W), maxshape = (total_images, H, W), dtype = 'float32', chunks = True)
    dset_names = f.create_dataset(f"{split}/names", shape = (0,), maxshape = (total_images,), dtype = h5py.string_dtype(), chunks = True)
    idx = 0
    for name in tqdm.tqdm(names, desc = f"Processing {split}"):
        image, gt, conf, att = process_image(dataset_path, name, sigma, bands)
        for dset, data in zip([dset_img, dset_gt, dset_conf, dset_att, dset_names], [image, gt, conf, att, name]):
            dset.resize((idx + 1,) + dset.shape[1:])
            dset[idx] = data
        idx += 1
        if augment:
            aug_imgs = augment_images(image)
            aug_gts = augment_images(gt[..., None])
            aug_confs = augment_images(conf)
            aug_atts = augment_images(att[..., None])
            for i in range(1, aug_imgs.shape[0]):
                aug_img = aug_imgs[i]
                aug_gt = np.squeeze(aug_gts[i], axis = -1)
                aug_conf = aug_confs[i]
                aug_att = np.squeeze(aug_atts[i], axis = -1)
                for dset, data in zip([dset_img, dset_gt, dset_conf, dset_att, dset_names], [aug_img, aug_gt, aug_conf, aug_att, name]):
                    dset.resize((idx + 1,) + dset.shape[1:])
                    dset[idx] = data
                idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help = 'path to dataset')
    parser.add_argument('output', help = 'output path for .h5 file')
    parser.add_argument('--train', default = 'train.txt', help = 'train split file name')
    parser.add_argument('--val', default = 'val.txt', help = 'validation split file name')
    parser.add_argument('--test', default = 'test.txt', help = 'test split file name')
    parser.add_argument('--augment', action = 'store_true', help = 'apply augmentation')
    parser.add_argument('--sigma', type = float, default = 3, help = 'Gaussian kernel size in pixels')
    parser.add_argument('--bands', default = 'RGBN', help = 'input raster bands (RGB or RGBN)')
    args = parser.parse_args()
    
    with h5py.File(args.output, 'w') as f:
        process_split(f, args.dataset, args.train, 'train', args.sigma, args.bands, augment = args.augment)
        process_split(f, args.dataset, args.val, 'val', args.sigma, args.bands, augment = False)
        process_split(f, args.dataset, args.test, 'test', args.sigma, args.bands, augment = False)
        f.attrs['bands'] = args.bands

if __name__ == '__main__':
    main()
