## Urban Tree Detection ##

This repository provides code for training and evaluating a convolutional neural network (CNN) to detect tree in urban environments with aerial imagery.   The CNN takes multispectral imagery as input and outputs a confidence map indicating the locations of trees. The individual tree locations are found by local peak finding. In our study site in Southern California, we determined that, using our trained model, 73.6% of the detected trees matched to actual trees, and 73.3% of the trees in the study area were detected.

### Installation ###

The model is implemented with Python 3.11.4 and TensorFlow 2.18.0.  We have provided an `environment.yml` file for earlier versions of Python and TensorFlow so that you can easily create a conda environment with the dependencies installed:

    conda env create 
    conda activate urban-tree-detection

For Python 3.11.4 and TensorFlow 2.18.0, you may run

    python -m venv env
    source env/bin/activate
    pip install tensorflow[and-cuda]==2.18.0
    pip install numpy
    pip install imageio
    pip install rasterio
    pip install geopandas
    pip install h5py
    pip install scipy
    pip install tqdm
    pip install scikit-image
    pip install scikit-learn
    pip install optuna
    pip install matplotlib

### Dataset ###

The data used in our paper can be found in [a separate Github repository](https://github.com/jonathanventura/urban-tree-detection-data/).

To prepare a dataset for training and testing, run the `prepare.py` script.  You can specify the bands in the input raster using the `--bands` flag (currently `RGB` and `RGBN` are supported.)

    python3 -m scripts.prepare <path to dataset> <path to hdf5 file> --bands <RGB or RGBN>

For example,

    python3 -m scripts.prepare ../urban-tree-detection-data prepared_data.hdf5

### Training ###

To train the model, run the `train.py` script.

    python3 -m scripts.train <path to hdf5 file> <path to log directory>

For example,

    python3 -m scripts.train prepared_data.hdf5 logs 

### Hyperparameter tuning ###

The model outputs a confidence map, and we use local peak finding to isolate individual trees.  We use the Optuna package to determine the optimal parameters of the peaking finding algorithm.  We search for the best of hyperparameters to maximize F-score on the validation set.

    python3 -m scripts.tune <path to hdf5 file> <path to log directory>

For example,

    python3 -m scripts.tune prepared_data.hdf5 logs

### Evaluation on test set ###

Once hyperparameter tuning finishes, use the `test.py` script to compute evaluation metrics on the test set.

    python3 -m scripts.test <path to hdf5 file> <path to log directory> --center_crop --rearrange_channels

For example,

    python3 -m scripts.test prepared_data.hdf5 logs

### Inference on a large raster ###

To detect trees in rasters and produce GeoJSONs containing the geo-referenced trees, use the `inference.py` script.  The script can process a single raster or a directory of rasters.

    python3 -m scripts.inference <input tiff or directory> <output json or directory> <path to log directory> --bands <RGB or RGBN>

For example,

    python3 -m scripts.inference ../urban-tree-detection-data/image_for_inference.tif inferred_geospatial_layer.json logs

### Pre-trained weights ###

The following pre-trained models are available:

| Imagery   | Years     | Bands    | Region                         | Log Directory Archive     |
|-----------|-----------|----------|--------------------------------|---------------------------|
| 60cm NAIP | 2016-2020 | RGBN     | Northern & Southern California | [OneDrive](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/ES31TXWdeGRFj_hn3O4qZpoBfhye_ssuULyaC2WB7yaJTw?e=cYkjMf) |
| 60cm NAIP | 2016-2020 | RGB      | Northern & Southern California | [OneDrive](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/Eay6v76obwpIqJmeK23_4zUBNb5EwM6R36wcSqh_BWKj_g?e=JrOwkO)
| 60cm NAIP | 2020      | RGBN     | Southern California            | [OneDrive](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/EQMSOBZjuDFCjj_PNgSDXZ0BMQUcGQKUO_SlJ5SGH2Bl9Q?e=9RhhpN)

We also provide an [example NAIP 2020 tile from Los Angeles](https://cpslo-my.sharepoint.com/:i:/g/personal/jventu09_calpoly_edu/EU1xfporUiBDvT2ZOpW0raEBOqJcJQpqcOv1lKNMCgbCdQ?e=zsgxXs) and an [example GeoJSON predictions file from the RGBN 2016-2020 model](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/EUHYGnWdqL5FvYc1wm9hSl8BBdL2JEgMSlqS1FiTdB0EWA?e=uZMIBc).  

You can explore a [map of predictions for the entire urban reserve of California](https://jventu09.users.earthengine.app/view/urban-tree-detector) (based on NAIP 2020 imagery) created using this pre-trained model.

### Using your own data ###

To train on your own data, you will need to organize the data into the format expected by `prepare.py`.

* The image crops (or "chips") should all be the same size and the side length should be a multiple of 32.
* The code is currently designed for three-band (RGB) or four-band (red, green, blue, near-IR) imagery.  To handle more bands, you would need to add an appropriate preprocessing function in `utils/preprocess.py`.  If RGB are not in the bands, then `models/VGG.py` would need to be modified, as the code expects the first three bands to be RGB to match the pre-trained weights.
* Store the images as TIFF or PNG files in a subdirectory called `images`.
* For each image, store a csv file containing x,y coordinates for the tree locations in a file `<name>.csv` where `<name>.tif`, `<name>.tiff`, or `<name>.png` is the corresponding image. The csv file should have a single header line.
* Create the files `train.txt`, `val.txt`, and `test.txt` to specify the names of the files in each split.

### Citation ###

If you use or build upon this repository, please cite our paper:

J. Ventura, C. Pawlak, M. Honsberger, C. Gonsalves, J. Rice, N.L.R. Love, S. Han, V. Nguyen, K. Sugano, J. Doremus, G.A. Fricker, J. Yost, and M. Ritter (2024). [Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery.](https://www.sciencedirect.com/science/article/pii/S1569843224002024)  International Journal of Applied Earth Observation and Geoinformation, 130, 103848.

### Acknowledgments ###

This project was funded by CAL FIRE (award number: 8GB18415) the US Forest Service (award number: 21-CS-11052021-201), and an incubation grant from the Data Science Strategic Research Initiative at California Polytechnic State University.
