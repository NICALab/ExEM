import os
import torch
import numpy as np
import tifffile
import torch.version
from torch.utils.data import DataLoader
from model.generator_3d_groupnorm import Anisotrpic_UNet3D as UNet
from src.utils.dataset import Light2EM_Dataset_test
from tqdm import tqdm


def create_bump_function(size, sigma=0.5):
    """
    Creates a 3D Gaussian bump function for a given patch size with adjustable sigma.
    The bump function value is highest at the center and decreases towards the edges.
    Sigma controls the steepness of the decline.
    """
    z, y, x = size
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, z, dtype=np.float32),
        np.linspace(-1, 1, y, dtype=np.float32),
        np.linspace(-1, 1, x, dtype=np.float32),
        indexing='ij'
    )
    bump = np.exp(-((zz**2 + yy**2 + xx**2) / (2 * sigma**2)))
    return bump


def inference(model, LM_data_path, patch_size, patch_overlap, batch_size, roi, sigma=0.5, device="cuda"):
    testset = Light2EM_Dataset_test(LM_data_path, patch_size, patch_overlap, roi)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True)

    result_array = np.zeros(testset.LM_data.shape, dtype=np.float32)
    weight_array = np.zeros(testset.LM_data.shape, dtype=np.float32)

    # Create bump function for the patch size
    bump_function = create_bump_function(patch_size, sigma=sigma)

    model.eval()
    for i, data in enumerate(tqdm(testloader)):
        x = data["A"]
        x = x.unsqueeze(1)
        x = x.to(device)
        x = x / 255.0
        coordinate = data["coordinate"]
        patch_coordinate = data["patch_coordinate"]
        coordinate_raw = data["coordinate_raw"]

        x_start = coordinate_raw[0]
        x_end = x_start + patch_size[2]
        y_start = coordinate_raw[1]
        y_end = y_start + patch_size[1]
        z_start = coordinate_raw[2]
        z_end = z_start + patch_size[0]

        with torch.no_grad():
            output = model(x)
            for bi in range(output.shape[0]):
                patch = output[bi, 0, :, :, :].cpu().numpy()
                result_array[z_start[bi]:z_end[bi], y_start[bi]:y_end[bi], x_start[bi]:x_end[bi]] += patch * bump_function
                weight_array[z_start[bi]:z_end[bi], y_start[bi]:y_end[bi], x_start[bi]:x_end[bi]] += bump_function
    
    # Normalize the result by the weight array
    result_array /= weight_array
    result_array = np.nan_to_num(result_array)  # Handle divisions by zero

    return result_array, testset


if __name__=="__main__":
    LM_data_path = "Your path to the LM data"
    model_file = "Your path to the trained model"
    output_file_dir = "Your output directory"
    patch_size = [16, 256, 256]
    patch_overlap = [8, 128, 128]
    batch_size = 8
    roi = [0, 64, 0, 1024, 0, 1024] # [z_start, z_end, y_start, y_end, x_start, x_end]
    sigma = 0.5  # Adjust this value to control the gradient of the bump function
    is_multi_gpu = False # Set to True if the model was trained with multiple GPUs

    # make output directory
    os.makedirs(output_file_dir, exist_ok=True)

    # check if the model file exists
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = UNet(n_channels=1, n_classes=1, num_channels=[32, 64, 128, 256, 512])
    
    # load model
    if is_multi_gpu:
        state_dict = torch.load(model_file)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(model_file))
    model = model.cuda()

    result_array, testset = inference(model, LM_data_path, patch_size, patch_overlap, batch_size, roi, sigma=sigma)