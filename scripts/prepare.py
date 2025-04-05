import argparse
import os
import imageio
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import tqdm
import cv2

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

# def augment_images(images):
#     '''
#     Return an augmented stack of 8 versions of the image, including 4 90 degree rotations and the vertical flip of each.
#     '''
#     augmented = []
#     for k in range(0, 4):
#         rotated = np.rot90(image, k = k)
#         augmented.append(rotated)
#         augmented.append(np.flipud(rotated))
#     return np.stack(augmented)

def augment_images(images, num_augmented=4):
    '''
    For each input image, generate a wide set of augmentations including:
    - 4 rotations (0°, 90°, 180°, 270°)
    - Vertical flips
    - Random crops
    - Brightness/contrast adjustment
    - Affine transforms (rotate/scale/translate)

    From the generated set, randomly select `num_augmented` augmentations per image.
    
    Parameters:
        images (numpy.ndarray): Stack of input images, shape (N, H, W, C)
        num_augmented (int): Number of augmentations to return per input image

    Returns:
        numpy.ndarray: Stack of augmented images, shape (N * num_augmented, H, W, C)
    '''
    output = []

    for image in images:
        augmented = []

        for k in range(4):
            rotated = np.rot90(image, k=k)
            augmented.append(rotated)
            augmented.append(np.flipud(rotated))

            # Random crop
            h, w = rotated.shape[:2]
            crop_scale = np.random.uniform(0.7, 1.0)
            new_h, new_w = int(h * crop_scale), int(w * crop_scale)
            y_start = np.random.randint(0, h - new_h + 1)
            x_start = np.random.randint(0, w - new_w + 1)
            cropped = rotated[y_start:y_start+new_h, x_start:x_start+new_w]
            cropped = cv2.resize(cropped, (w, h))
            augmented.append(cropped)

            # Brightness/Contrast
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-30, 30)    # brightness
            bright_contrast = np.clip(alpha * rotated + beta, 0, 255).astype(np.uint8)
            augmented.append(bright_contrast)

            # Affine Transform
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2),
                                             np.random.uniform(-45, 45),
                                             np.random.uniform(0.9, 1.2))
            affine = cv2.warpAffine(rotated, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            if image.shape[-1] == 1:
                affine = affine[..., np.newaxis]
            elif affine.ndim == 2:
                affine = np.stack([affine]*image.shape[-1], axis=-1)
            augmented.append(affine)

        # Randomly select `num_augmented` variants from the augmented list
        np.random.seed(0)
        selected = np.random.choice(len(augmented), size=num_augmented, replace=False)
        for idx in selected:
            output.append(augmented[idx])

    return np.stack(output)

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
