# Training package for the BBHnet

This package contains scripts and tutorial to train a BBHnet model on
a preprocessed dataset (note: the package will not apply preprocess by itself).

Use `train_bbhnet.py` to run training.
The dataset directory has to have the following format:

```
dataset_dir/
    train/
        n00.h5
        n01.h5
        ...
    val/
        n00.h5
        n01.h5
        ...
```

where the `nXX.h5` are HDF5 with the following fields:

- `data`:
Strain data of shape `(N, 2, 1024)` where `N` is the number of samples
In the second dimension, the first channel is always Hanford strain,
and the second channel is always Livingston.

- `corr`:
The Pearson correlation array between Hanford and Livingston of each sample.
Must be in shape of `(N, corr_dim`) where `corr_dim` is the length of the correlation array.

- `label`:
The label of each sample in shape of `(N, 1)`.
The label can be `0` (noise) or `1` (signal).
In some cases, the `glitches` class is included and labeled `2`.
In these cases, `train_bbhnet.py` will automatically convert all `2` labels to `0`.

Arguments of `train_bbhnet.py`:
```
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path to input dataset directory
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to output directory
  --store-val-output    Enable to store validation data NN output to outdir
  -A {CNN-SMALL,CNN-MEDIUM,CNN-LARGE,FC-CORR}, --arch {CNN-SMALL,CNN-MEDIUM,CNN-LARGE,FC-CORR}
                        NN architecture to use
  --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Input dimension for NN, excluding batch dimension. Default is (2, 1024)
  --corr-dim CORR_DIM   Dimension of correlation array. If 0 (default) is given, correlation will not be used
  -e MAX_EPOCHS, --max-epochs MAX_EPOCHS
                        Maximum number of epochs
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate of ADAM optimizer
  -l1 WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        L1 regularization of ADAM optimizer
  --shuffle             Enable to shuffle training data
  --resume              Resume training using the latest model available
```

