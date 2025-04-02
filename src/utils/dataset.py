import math
import zarr
import numpy as np
import torch
import torchio as tio

from torch.utils.data import Dataset

def random_transform(input, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [Z, X, Y])
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [Z, X, Y])
    """

    # Random flip along each axis
    if np.random.rand() > 0.5:  # Flip along Z-axis
        input = torch.flip(input, dims=[0])  # Flip along first dimension (Z)

    if np.random.rand() > 0.5:  # Flip along X-axis
        input = torch.flip(input, dims=[1])  # Flip along second dimension (X)

    if np.random.rand() > 0.5:  # Flip along Y-axis
        input = torch.flip(input, dims=[2])  # Flip along third dimension (Y)

    # Random rotation by 90 degrees if is_rotate is True
    if is_rotate:
        num_rotations = np.random.randint(0, 4)  # Rotate by 0, 90, 180, or 270 degrees
        input = torch.rot90(input, k=num_rotations, dims=[1, 2])  # Rotate in the XY plane

    return input


def get_coordinate(img_size, patch_size, patch_interval):
    """DeepCAD version of stitching
    https://github.com/cabooster/DeepCAD/blob/53a9b8491170e298aa7740a4656b4f679ded6f41/DeepCAD_pytorch/data_process.py#L374
    """
    whole_s, whole_h, whole_w = img_size
    img_s, img_h, img_w = patch_size
    gap_s, gap_h, gap_w = patch_interval

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    # print(whole_s, whole_h, whole_w)
    # print(img_s, img_h, img_w)
    # print(gap_s, gap_h, gap_w)

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)

    coordinate_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    if num_w > 1:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    else:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    if num_h > 1:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    else:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = x*gap_h+img_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    if num_s > 1:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s-cut_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                coordinate_list.append(single_coordinate)

    return coordinate_list


class Light2EM_Dataset(Dataset):
    def __init__(self, LM_data_path, EM_data_path, patch_size, overlap_size, concat_LM=None, balanced_sampling=True):
        """
        Args:
            LM_data_path (list): list of zarr file paths for LM data
            EM_data_path (list): list of zarr file paths for EM data
            patch_size (list): [z, y, x]
            overlap_size (list): [z, y, x]
            concat_LM (list): list of zarr file paths for concatenated LM data
            balanced_sampling (bool): whether to use balanced sampling
        """

        if not balanced_sampling:
            raise NotImplementedError("Currently only support balanced sampling")
        
        self.concat_LM = concat_LM
        if concat_LM:
            self.LM_data = []
            self.LM_data_pair = []
            for LM_file_1, LM_file_2 in zip(LM_data_path, concat_LM):
                LM_data_1 = zarr.open(LM_file_1, mode='r')
                self.LM_data.append(LM_data_1)
                LM_data_2 = zarr.open(LM_file_2, mode='r')
                self.LM_data_pair.append(LM_data_2)
                # check the pair
                assert LM_data_1.shape == LM_data_2.shape, "LM data shape mismatch"
                print("LM {}: (z, y, x) = {}".format(LM_file_1, LM_data_1.shape))
                print("LM {}: (z, y, x) = {}".format(LM_file_2, LM_data_2.shape))
                print("----------------")
        else:
            self.LM_data = []
            for LM_file in LM_data_path:
                LM_data = zarr.open(LM_file, mode='r')
                self.LM_data.append(LM_data)
                print("LM shape: (z, y, x) = {}".format(LM_data.shape))
        self.EM_data = []
        for EM_file in EM_data_path:
            EM_data = zarr.open(EM_file, mode='r')
            self.EM_data.append(EM_data)
            print("EM shape: (z, y, x) = {}".format(EM_data.shape))
        self.patch_size = patch_size
        self.overlap_size = overlap_size

        # get patch indices
        self.LM_patch_indices = []
        self.LM_patch_num = 0
        self.EM_patch_indices = []
        self.EM_patch_num = 0

        for _, current_LM_data in enumerate(self.LM_data):
            current_ds_LM_patch_indices = []
            for z in range(0, current_LM_data.shape[0], patch_size[0]-overlap_size[0]):
                for y in range(0, current_LM_data.shape[1], patch_size[1]-overlap_size[1]):
                    for x in range(0, current_LM_data.shape[2], patch_size[2]-overlap_size[2]):
                        z_idx = z
                        y_idx = y
                        x_idx = x
                        if z+patch_size[0] > current_LM_data.shape[0]:
                            z_idx = current_LM_data.shape[0]-patch_size[0]
                        if y+patch_size[1] > current_LM_data.shape[1]:
                            y_idx = current_LM_data.shape[1]-patch_size[1]
                        if x+patch_size[2] > current_LM_data.shape[2]:
                            x_idx = current_LM_data.shape[2]-patch_size[2]
                        current_ds_LM_patch_indices.append([z_idx, y_idx, x_idx])
                        self.LM_patch_num += 1
                        if (x_idx != x) or x_idx == current_LM_data.shape[2]-patch_size[2]:
                            break
                    if (y_idx != y) or y_idx == current_LM_data.shape[1]-patch_size[1]:
                        break
                if (z_idx != z) or z_idx == current_LM_data.shape[0]-patch_size[0]:
                    break
            self.LM_patch_indices.append(current_ds_LM_patch_indices)

        for _, current_EM_data in enumerate(self.EM_data):
            current_ds_EM_patch_indices = []
            for z in range(0, current_EM_data.shape[0], patch_size[0]-overlap_size[0]):
                for y in range(0, current_EM_data.shape[1], patch_size[1]-overlap_size[1]):
                    for x in range(0, current_EM_data.shape[2], patch_size[2]-overlap_size[2]):
                        z_idx = z
                        y_idx = y
                        x_idx = x
                        if z+patch_size[0] > current_EM_data.shape[0]:
                            z_idx = current_EM_data.shape[0]-patch_size[0]
                        if y+patch_size[1] > current_EM_data.shape[1]:
                            y_idx = current_EM_data.shape[1]-patch_size[1]
                        if x+patch_size[2] > current_EM_data.shape[2]:
                            x_idx = current_EM_data.shape[2]-patch_size[2]
                        current_ds_EM_patch_indices.append([z_idx, y_idx, x_idx])
                        self.EM_patch_num += 1
                        if (x_idx != x) or x_idx == current_EM_data.shape[2]-patch_size[2]:
                            break
                    if (y_idx != y) or y_idx == current_EM_data.shape[1]-patch_size[1]:
                        break
                if (z_idx != z) or z_idx == current_EM_data.shape[0]-patch_size[0]:
                    break
            self.EM_patch_indices.append(current_ds_EM_patch_indices)
        
        print("Number of total LM patches: {}".format(len(self.LM_patch_indices)))
        print("Number of total EM patches: {}".format(len(self.EM_patch_indices)))
    
    def __len__(self):
        max_val = max(self.LM_patch_num, self.EM_patch_num)
        return max_val

    def __getitem__(self, idx):
        LM_dataset_idx = idx % len(self.LM_patch_indices)
        EM_dataset_idx = idx % len(self.EM_patch_indices)

        LM_idx = (idx // len(self.LM_patch_indices)) % len(self.LM_patch_indices[LM_dataset_idx])
        EM_idx = (idx // len(self.EM_patch_indices)) % len(self.EM_patch_indices[EM_dataset_idx])

        z, y, x = self.LM_patch_indices[LM_dataset_idx][LM_idx]
        if self.concat_LM:
            LM_patch_1 = self.LM_data[LM_dataset_idx][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            LM_patch_2 = self.LM_data_pair[LM_dataset_idx][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            LM_patch = np.stack((LM_patch_1, LM_patch_2), axis=0)
        else:
            LM_patch = self.LM_data[LM_dataset_idx][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            LM_patch = torch.from_numpy(LM_patch).to(torch.float32)
            LM_patch = random_transform(LM_patch)
            LM_patch = torch.unsqueeze(LM_patch, axis=0)

        z, y, x = self.EM_patch_indices[EM_dataset_idx][EM_idx]
        EM_patch = self.EM_data[EM_dataset_idx][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
        EM_patch = torch.from_numpy(EM_patch).to(torch.float32)
        EM_patch = random_transform(EM_patch)
        EM_patch = torch.unsqueeze(EM_patch, axis=0)

        return {"A": LM_patch, "B": EM_patch}


class Light2EM_Dataset_test(Dataset):
    def __init__(self, LM_data_path, patch_size, overlap_size, roi=None):
        self.LM_data = zarr.open(LM_data_path, mode='r')
        if roi:
            self.LM_data = self.LM_data[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        self.patch_size = patch_size
        self.overlap_size = overlap_size

        print("LM shape: (z, y, x) = {}".format(self.LM_data.shape))

        self.LM_patch_indices = []

        for z in range(0, self.LM_data.shape[0], patch_size[0]-overlap_size[0]):
            for y in range(0, self.LM_data.shape[1], patch_size[1]-overlap_size[1]):
                for x in range(0, self.LM_data.shape[2], patch_size[2]-overlap_size[2]):
                    z_idx = z
                    y_idx = y
                    x_idx = x
                    if z+patch_size[0] > self.LM_data.shape[0]:
                        z_idx = self.LM_data.shape[0]-patch_size[0]
                    if y+patch_size[1] > self.LM_data.shape[1]:
                        y_idx = self.LM_data.shape[1]-patch_size[1]
                    if x+patch_size[2] > self.LM_data.shape[2]:
                        x_idx = self.LM_data.shape[2]-patch_size[2]
                    self.LM_patch_indices.append([z_idx, y_idx, x_idx])
                    if (x_idx != x) or x_idx == self.LM_data.shape[2]-patch_size[2]:
                        break
                if (y_idx != y) or y_idx == self.LM_data.shape[1]-patch_size[1]:
                    break
            if (z_idx != z) or z_idx == self.LM_data.shape[0]-patch_size[0]:
                break
        
        print("Number of LM patches: {}".format(len(self.LM_patch_indices)))

    def __len__(self):
        return len(self.LM_patch_indices)
    
    def __getitem__(self, idx):
        single_coordinate = self.LM_patch_indices[idx]

        z, y, x = single_coordinate
        
        # z change
        if z == 0:
            z_start = 0
            z_patch_start = 0
        else:
            z_start = z + self.overlap_size[0]//2
            z_patch_start = self.overlap_size[0]//2
        if z == self.LM_data.shape[0]-self.patch_size[0]:
            z_end = self.LM_data.shape[0]
            z_patch_end = self.patch_size[0]
        else:
            z_end = z + self.patch_size[0] - self.overlap_size[0]//2
            z_patch_end = self.patch_size[0] - self.overlap_size[0]//2

        # y change
        if y == 0:
            y_start = 0
            y_patch_start = 0
        else:
            y_start = y + self.overlap_size[1]//2
            y_patch_start = self.overlap_size[1]//2
        if y == self.LM_data.shape[1]-self.patch_size[1]:
            y_end = self.LM_data.shape[1]
            y_patch_end = self.patch_size[1]
        else:
            y_end = y + self.patch_size[1] - self.overlap_size[1]//2
            y_patch_end = self.patch_size[1] - self.overlap_size[1]//2
        
        # x change
        if x == 0:
            x_start = 0
            x_patch_start = 0
        else:
            x_start = x + self.overlap_size[2]//2
            x_patch_start = self.overlap_size[2]//2
        if x == self.LM_data.shape[2]-self.patch_size[2]:
            x_end = self.LM_data.shape[2]
            x_patch_end = self.patch_size[2]
        else:
            x_end = x + self.patch_size[2] - self.overlap_size[2]//2
            x_patch_end = self.patch_size[2] - self.overlap_size[2]//2
        
        LM_patch = self.LM_data[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
        LM_patch = torch.from_numpy(LM_patch).to(torch.float32)

        return {"A": LM_patch, "coordinate": (x_start, x_end, y_start, y_end, z_start, z_end),\
         "patch_coordinate": (x_patch_start, x_patch_end, y_patch_start, y_patch_end, z_patch_start, z_patch_end), "coordinate_raw": (x, y, z)}


class Light2EM_Dataset_test_efficient(Dataset):
    def __init__(self, LM_data_path, patch_size, overlap_size, roi=None):
        self.LM_data_path = LM_data_path
        self.roi = roi
        self.patch_size = patch_size
        self.overlap_size = overlap_size

        # Open zarr in read mode to check its dimensions, then close
        LM_data = zarr.open(self.LM_data_path, mode='r')
        self.LM_data = LM_data
        if roi:
            self.data_shape = (
                roi[1] - roi[0],
                roi[3] - roi[2],
                roi[5] - roi[4]
            )
        else:
            self.data_shape = LM_data.shape

        print("LM shape: (z, y, x) = {}".format(self.data_shape))

        self.LM_patch_indices = []
        for z in range(0, self.data_shape[0], patch_size[0] - overlap_size[0]):
            for y in range(0, self.data_shape[1], patch_size[1] - overlap_size[1]):
                for x in range(0, self.data_shape[2], patch_size[2] - overlap_size[2]):
                    z_idx = z
                    y_idx = y
                    x_idx = x
                    if z + patch_size[0] > self.data_shape[0]:
                        z_idx = self.data_shape[0] - patch_size[0]
                    if y + patch_size[1] > self.data_shape[1]:
                        y_idx = self.data_shape[1] - patch_size[1]
                    if x + patch_size[2] > self.data_shape[2]:
                        x_idx = self.data_shape[2] - patch_size[2]
                    self.LM_patch_indices.append([z_idx, y_idx, x_idx])
                    if (x_idx != x) or x_idx == self.data_shape[2] - patch_size[2]:
                        break
                if (y_idx != y) or y_idx == self.data_shape[1] - patch_size[1]:
                    break
            if (z_idx != z) or z_idx == self.data_shape[0] - patch_size[0]:
                break

        print("Number of LM patches: {}".format(len(self.LM_patch_indices)))

    def __len__(self):
        return len(self.LM_patch_indices)
    
    def __getitem__(self, idx):
        # Access specific patch coordinates
        single_coordinate = self.LM_patch_indices[idx]
        z, y, x = single_coordinate
        
        # Apply ROI if provided
        if self.roi:
            z_start = self.roi[0] + z
            y_start = self.roi[2] + y
            x_start = self.roi[4] + x
        else:
            z_start, y_start, x_start = z, y, x

        z_end = z_start + self.patch_size[0]
        y_end = y_start + self.patch_size[1]
        x_end = x_start + self.patch_size[2]

        # Retrieve patch
        LM_patch = self.LM_data[z_start:z_end, y_start:y_end, x_start:x_end]

        # Convert to torch tensor
        LM_patch = torch.from_numpy(LM_patch).to(torch.float32)

        # Calculate actual and patch coordinates based on overlap
        z_patch_start, z_patch_end = 0, self.patch_size[0]
        y_patch_start, y_patch_end = 0, self.patch_size[1]
        x_patch_start, x_patch_end = 0, self.patch_size[2]

        return {
            "A": LM_patch,
            "coordinate": (x_start, x_end, y_start, y_end, z_start, z_end),
            "patch_coordinate": (x_patch_start, x_patch_end, y_patch_start, y_patch_end, z_patch_start, z_patch_end),
            "coordinate_raw": (x, y, z)
        }
    

if __name__=="__main__":
    if False:
        dataset_3D = Light2EM_Dataset("./total_image.zarr", "./total_image_EM.zarr", [32, 512, 512], [16, 256, 256])
        # dataset_2D = Light2EM_Dataset("./total_image.zarr", "./total_image_EM.zarr", [1, 512, 512], [0, 256, 256])

        imgs = dataset[0]
        LM_patch = imgs["A"]
        EM_patch = imgs["B"]
        print(LM_patch.shape)
        print(max(LM_patch.flatten()))
        print(EM_patch.shape)
        print(max(EM_patch.flatten()))
    if True:
        dataset_3D_test = Light2EM_Dataset_test("", [16, 256, 256], [8, 128, 128])
        data = dataset_3D_test[1]
        LM_patch = data["A"]
        coordinate = data["coordinate"]
        print(LM_patch.shape)
        print(coordinate)
