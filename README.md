# ExEM: Automated analysis of densely labeled expansion microscopy images using electron microscopy resources

Understanding cellular mechanisms requires nanoscale imaging that simultaneously captures molecular specificity and ultrastructural context. Expansion microscopy (ExM) with antibody and pan-protein staining offers a promising alternative to electron microscopy (EM). However, ExM images lack the sharp contrast and structural clarity of EM, complicating interpretation and limiting compatibility with the extensive computational ecosystem developed for EM-based analysis. Here, we introduce ExEM (Expansion-enabled virtual Electron Microscopy), a self-supervised learning–based framework that transforms ExM images into virtual EM images with enhanced contrast and clearly defined structural boundaries. By learning from EM data, ExEM facilitates intuitive interpretation and bridges the modality gap, allowing ExM datasets to be analyzed using pre-existing EM-trained segmentation models without manual annotation or model retraining. This approach enables automated, high-throughput analysis of densely labeled ExM volumes using EM-style tools and resources, thereby unlocking the full analytical potential of ExM across domains such as neuroscience, cell biology, and cancer research. 

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
git clone https://github.com/NICALab/ExEM.git
cd ExEM
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
