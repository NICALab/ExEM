# VCM: Expansion microscopy enables virtual correlative super-resolution optical and electron microscopy

Understanding cellular mechanisms requires visualization at nanoscale of biomolecules and their structural context. While electron microscopy (EM) provides ultrastructural details, it lacks multiplexing capability and requires specialized expertise. Correlative light-electron microscopy addresses some limitations but struggles with registration and resolution misalignments. Here, we introduce Virtual Correlative super-resolution optical and electron Microscopy (VCM), an imaging pipeline that integrates 16× expansion microscopy (ExM) with deep-learning-based virtual EM generation. VCM fully leverages the potential of ExM to achieve high-resolution visualization of both biomolecules and ultrastructure with EM-level contrast, using only conventional fluorescence microscopy. Another key feature of VCM is its ability to generate virtual EM images compatible with existing EM resources, allowing direct application of tools such as pre-trained segmentation models. This significantly reduces the cost and effort for segmentation, accelerating deep-learning-based ultrastructural analysis. 

## Overview

This repository implements a GAN-based model for the image translation task between LM and EM domains. Key features include:

- **3D Patch-Based Training:** The model is trained on 3D patches extracted from large LM and EM volumes.
- **Anisotropic 3D U-Net Generator:** A modified U-Net with group normalization that is tailored to handle anisotropic imaging data.
- **Multi-Scale Discriminator:** A multi-scale discriminator network that processes the generated images at different resolutions.
- **Distributed Training Support:** Training supports multi-GPU distributed environments using PyTorch’s distributed package.
- **Inference with Patch Stitching:** The test script employs a bump function to smoothly stitch the overlapping patch outputs back into a full volume.


## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/) (with CUDA support)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [Zarr](https://zarr.readthedocs.io/)
- [TorchIO](https://torchio.readthedocs.io/)
- [scikit-image](https://scikit-image.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard) (for logging)

Install the required packages using pip:

```bash
pip install torch numpy tqdm zarr torchio scikit-image tensorboard
```

## Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/NICALab/VCM.git
cd VCM
```

Ensure that your directory structure matches the one outlined above.

## Training

To train the model, run the training script with the desired configuration. The training code supports distributed training if multiple GPUs are available.

Example command for single GPU training:

```bash
python train.py --LM_data_path /path/to/LM/data --EM_data_path /path/to/EM/data --exp_name my_experiment --n_epochs 500 --batch_size 1 --lr 0.0003 --patch_size 1 512 512 --patch_overlap 0 256 256
```

The `parse_arguments` function (in `src/utils/util.py`) defines numerous configurable options for:
- Experiment naming and checkpointing.
- Dataset paths and patch extraction parameters.
- Loss weights (cycle loss, content loss, GAN loss, etc.).
- Learning rate and optimizer configurations.
- Optional scheduler and gradient clipping settings.

Training logs and generated images are saved to the specified `results_dir`.

## Testing / Inference

The test script (`test.py`) performs inference on LM volumes by:
- Loading the trained generator model.
- Extracting overlapping patches using a predefined region of interest (ROI).
- Applying a Gaussian “bump” function for smooth patch stitching.
- Reconstructing the full volume from the processed patches.


Adjust the parameters (patch size, overlap, ROI, etc.) according to your dataset and experimental requirements.

## Model Architecture

### Generator

The generator is implemented as an anisotropic 3D U-Net with group normalization (see `model/generator_3d_groupnorm.py`). It consists of:

- A first convolution layer to increase the channel dimension.
- A series of triple convolution blocks with downsampling and upsampling layers.
- Skip connections between corresponding down- and up-sampling stages.
- A final convolution layer that maps the output to the desired number of classes (channels).

### Discriminator

The discriminator (in `model/discriminator_3d.py`) is a multi-scale 3D network. Each scale is designed to process features at different resolutions. The model progressively downsamples the input volume and applies convolutional blocks with instance normalization and LeakyReLU activations.

## Dataset Preparation

The datasets are expected to be stored as Zarr files. The dataset class (`Light2EM_Dataset` in `src/utils/dataset.py`) handles:

- Reading LM and EM volumes from Zarr files.
- Extracting 3D patches with configurable sizes and overlaps.
- Applying random transformations (flipping and rotations) to augment the training data.

For testing, a similar dataset class (`Light2EM_Dataset_test`) is provided for efficient patch extraction and reconstruction.
