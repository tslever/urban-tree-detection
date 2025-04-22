import argparse
import os
import imageio
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import tqdm
import random
import tensorflow as tf

def process_image(dataset_path, name, sigma, bands, data_mode):
    image = None
    name_of_folder_of_images = 'images' if data_mode == 'full' else 'images_small'
    for suffix in ['.tif', '.tiff', '.png']:
        image_path = os.path.join(dataset_path, name_of_folder_of_images, name + suffix)
        if os.path.exists(image_path):
            image = imageio.imread(image_path)
            if image.ndim == 3 and image.shape[0] in [3, 4]:
                image = image.transpose(1, 2, 0)
            if suffix == '.png' or bands == 'RGB':
                image = image[..., :3]
            break
    if image is None:
        raise RuntimeError(f'Could not find image for {name}')

    name_of_folder_of_csv_files = 'csv' if data_mode == 'full' else 'csv_small'
    csv_path = os.path.join(dataset_path, name_of_folder_of_csv_files, name + '.csv')
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

def augment_images(image, seed, keep_dims=False, apply_color_aug=False):
    def to_3d(image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return image

    def from_3d(image):
        if not keep_dims and image.shape[2] == 1:
            return image[..., 0]  # Back to 2D
        return image

    def contrast(image):
        image = to_3d(image)
        image = tf.image.adjust_contrast(tf.convert_to_tensor(image, dtype=tf.float32), contrast_factor=0.8)
        return from_3d(image.numpy())

    def brightness(image):
        image = to_3d(image)
        image = tf.image.adjust_brightness(tf.convert_to_tensor(image, dtype=tf.float32), delta=0.2)
        return from_3d(image.numpy())

    def crop(image):
        image = to_3d(image)
        image = tf.image.resize(image, [256, 256])
        image = tf.cast(image, tf.float32)
        cropped = tf.image.random_crop(image, size=[224, 224, image.shape[2]])
        return from_3d(tf.image.resize(cropped, [256, 256]).numpy())

    def flip(image):
        image = to_3d(image)
        flipped = tf.image.flip_left_right(image).numpy()
        return from_3d(flipped)

    def rotate(image, k):
        image = to_3d(image)
        return from_3d(np.rot90(image, k=k))

    augmented = []

    if seed == 0:
        for k in range(0, 2):
            rotated = rotate(image, k)
            augmented.append(rotated)
            augmented.append(flip(rotated))
    elif seed == 1:
        augmented.append(crop(image))
        if apply_color_aug:
            augmented.append(brightness(image))
            augmented.append(contrast(image))
        augmented.append(flip(image))
    elif seed == 2:
        augmented.append(crop(image))
        augmented.append(rotate(image, k=1))
        if apply_color_aug:
            augmented.append(contrast(image))
        augmented.append(flip(image))
    elif seed == 3:
        augmented.append(rotate(image, k=3))
        if apply_color_aug:
            augmented.append(brightness(image))
            augmented.append(contrast(image))
        augmented.append(flip(image))
    elif seed == 4:
        augmented.append(crop(image))
        if apply_color_aug:
            augmented.append(brightness(image))
            augmented.append(contrast(image))
        augmented.append(rotate(image, k=1))
    elif seed == 5:
        augmented.append(crop(image))
        if apply_color_aug:
            augmented.append(brightness(image))
        augmented.append(rotate(image, k=3))
        augmented.append(flip(image))
    elif seed == 6:
        augmented.append(rotate(image, k=1))
        augmented.append(rotate(image, k=3))
        if apply_color_aug:
            augmented.append(contrast(image))
            augmented.append(brightness(image))

    return np.stack(augmented)

def process_split(f, dataset_path, split_file, split, sigma, bands, data_mode, augment = False):
    with open(os.path.join(dataset_path, split_file), 'r') as sf:
        names = [line.strip() for line in sf]
    sample_img, sample_gt, sample_conf, sample_att = process_image(dataset_path, names[0], sigma, bands, data_mode)
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
        image, gt, conf, att = process_image(dataset_path, name, sigma, bands, data_mode)
        for dset, data in zip([dset_img, dset_gt, dset_conf, dset_att, dset_names], [image, gt, conf, att, name]):
            dset.resize((idx + 1,) + dset.shape[1:])
            dset[idx] = data
        idx += 1
	if augment:
	    seed = random.randint(0, 6)
	    aug_imgs = augment_images(image, seed, apply_color_aug=True)
	    aug_gts = augment_images(gt, seed, keep_dims=False, apply_color_aug=False)
	    aug_confs = augment_images(conf, seed, keep_dims=True, apply_color_aug=False)
	    aug_atts = augment_images(att, seed, keep_dims=False, apply_color_aug=False)
	    for i in range(1, 4):  # Skip the first since it’s already added
	        aug_img = aug_imgs[i]
	        aug_gt = aug_gts[i]
	        aug_conf = aug_confs[i]
	        aug_att = aug_atts[i]
	        for dset, data in zip([dset_img, dset_gt, dset_conf, dset_att, dset_names],
	                              [aug_img, aug_gt, aug_conf, aug_att, name]):
	            dset.resize((idx + 1,) + dset.shape[1:])
	            dset[idx] = data
	        idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help = 'path to dataset')
    parser.add_argument('output', help = 'output path for .h5 file')
    parser.add_argument('--data_mode', choices = ['full', 'small'], default = 'full', help = 'Data mode: full uses normal csv, images, train.txt, etc.; small uses csv_small, images_small, train_small.txt, etc.')
    parser.add_argument('--train', default = None, help = 'train split file name')
    parser.add_argument('--val', default = None, help = 'validation split file name')
    parser.add_argument('--test', default = None, help = 'test split file name')
    parser.add_argument('--augment', action = 'store_true', help = 'apply augmentation')
    parser.add_argument('--sigma', type = float, default = 3, help = 'Gaussian kernel size in pixels')
    parser.add_argument('--bands', default = 'RGBN', help = 'input raster bands (RGB or RGBN)')
    args = parser.parse_args()
    
    if args.data_mode == 'small':
        train_file = 'train_small.txt' if args.train is None else args.train
        val_file = 'val_small.txt' if args.val is None else args.val
        test_file = 'test_small.txt' if args.test is None else args.test
    else:
        train_file = 'train.txt' if args.train is None else args.train
        val_file = 'val.txt' if args.val is None else args.val
        test_file = 'test.txt' if args.test is None else args.test
    
    with h5py.File(args.output, 'w') as f:
        process_split(f, args.dataset, train_file, 'train', args.sigma, args.bands, args.data_mode, augment = args.augment)
        process_split(f, args.dataset, val_file, 'val', args.sigma, args.bands, args.data_mode, augment = False)
        process_split(f, args.dataset, test_file, 'test', args.sigma, args.bands, args.data_mode, augment = False)
        f.attrs['bands'] = args.bands

if __name__ == '__main__':
    main()
