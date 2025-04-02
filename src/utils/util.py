import os
import math
import argparse
import numpy as np
import torch
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for rng")
    parser.add_argument("--effective_iter", type=int, default=0, help="effective_iter to start training from (need effective_iter model)")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--exp_name", type=str, default="myEXP", help="name of the experiment")
    parser.add_argument("--results_dir", type=str, default="./results", help="root directory to save results")

    # dataset
    parser.add_argument("--LM_data_path", type=str, nargs="+", help="path to the LM data")
    parser.add_argument("--LM_data_path_paired", type=str, nargs="+", help="path to the pair LM data")
    parser.add_argument("--EM_data_path", type=str, nargs="+", help="path to the EM data")
    parser.add_argument("--patch_size", type=int, default=[1, 512, 512], nargs="+", help="size of the patches")
    parser.add_argument("--patch_overlap", type=int, default=[0, 256, 256], nargs="+", help="size of the patch overlap")

    # training
    parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")
    parser.add_argument("--lr_D", type=float, default=3e-4, help="adam: learning rate for discriminator")
    parser.add_argument("--lambda_gen", type=float, default=1.0, help="generator loss weight")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_content", type=float, default=50.0, help="content loss weight")
    parser.add_argument("--lambda_vgg", type=float, default=1.0, help="vgg loss weight")
    parser.add_argument("--lambda_aug", type=float, default=1.0, help="augmentation consistency loss weight")
    parser.add_argument("--lambda_affinity", type=float, default=5.0, help="affinity loss weight")
    parser.add_argument("--lambda_GAN_affinity", type=float, default=1.0, help="GAN affinity loss weight")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--content_loss_A_threshold", type=float, default=110, help="threshold for content loss weight")
    parser.add_argument("--content_loss_A_pair_threshold", type=float, default=110, help="threshold for content loss weight")
    parser.add_argument("--content_loss_B_threshold", type=float, default=110, help="threshold for content loss weight")
    parser.add_argument("--content_loss_decay_batch", type=float, default=1000, help="decay rate of content loss weight (in batches)")
    parser.add_argument("--unet_num_first_channel", type=int, default=32, help="number of first channel in UNet")
    parser.add_argument("--discriminator_num_scales", type=int, default=2, help="number of scales in discriminator")
    parser.add_argument("--batch_norm_momentum", type=float, default=0.1, help="batch norm momentum")
    parser.add_argument("--gradient_max_norm", type=float, default=0, help="max norm of gradients")
    parser.add_argument("--replay_buffer_size", type=int, default=0, help="size of the replay buffer")
    parser.add_argument("--use_scheduler", action="store_true", help="use learning rate scheduler")
    parser.add_argument('--generator_channels', type=int, default=[32, 64, 128, 256, 512], nargs="+", help='channels of the generator')
    parser.add_argument('--discriminator_channels', type=int, default=64, help='channels of the discriminator')
    parser.add_argument('--use_AdamW', action="store_true", help="use AdamW optimizer")
    parser.add_argument('--use_AMP', action="store_true", help="use Automatic Mixed Precision")
    parser.add_argument("--model_parallelism", action="store_true", help="use model parallelism")

    # bottleneck parameters
    parser.add_argument("--use_total_bottleneck", action="store_true", help="use total bottleneck")

    # util
    parser.add_argument("--logging_interval_batch", type=int, default=50, help="interval between logging info (in batches)")
    parser.add_argument("--logging_interval", type=int, default=1, help="interval between logging info (in epochs)")
    parser.add_argument("--image_save_interval_batch", type=int, default=1000, help="interval between saving generated images (in batches)") # 1000: 10min
    parser.add_argument("--checkpoint_interval_batch", type=int, default=6000, help="interval between saving trained models (in epochs)") # 6000: 1hr
    opt = parser.parse_args()
    
    # check if LM_data_path is folder or zarr file
    total_LM_data_path = []
    for i in range(len(opt.LM_data_path)):
        if os.path.splitext(opt.LM_data_path[i])[1] != ".zarr": # if it is folder
            zarr_file_names = os.listdir(opt.LM_data_path[i])
            zarr_file_names = sorted(zarr_file_names)
            for zarr_file_name in zarr_file_names:
                if "zarr" in zarr_file_name:
                    total_LM_data_path.append(os.path.join(opt.LM_data_path[i], zarr_file_name))
        else:
            total_LM_data_path.append(opt.LM_data_path[i])
    opt.LM_data_path = total_LM_data_path
    
    # check if LM_data_path_paired is folder or zarr file
    if opt.LM_data_path_paired is not None:
        if not "zarr" in opt.LM_data_path_paired[0]:
            tmp_LM_data_path_paired = []
            zarr_file_names = os.listdir(opt.LM_data_path_paired[0])
            # sort
            zarr_file_names = sorted(zarr_file_names)
            for zarr_file_name in zarr_file_names:
                if "zarr" in zarr_file_name:
                    tmp_LM_data_path_paired.append(os.path.join(opt.LM_data_path_paired[0], zarr_file_name))
            opt.LM_data_path_paired = tmp_LM_data_path_paired
    
    # check if EM_data_path is folder or zarr file
    total_EM_data_path = []
    for i in range(len(opt.EM_data_path)):
        if os.path.splitext(opt.EM_data_path[i])[1] != ".zarr": # if it is folder
            zarr_file_names = os.listdir(opt.EM_data_path[i])
            zarr_file_names = sorted(zarr_file_names)
            for zarr_file_name in zarr_file_names:
                if "zarr" in zarr_file_name:
                    total_EM_data_path.append(os.path.join(opt.EM_data_path[i], zarr_file_name))
        else:
            total_EM_data_path.append(opt.EM_data_path[i])
    opt.EM_data_path = total_EM_data_path
    
    print(opt.LM_data_path)
    print(opt.EM_data_path)
    print("Length of LM data path: ", len(opt.LM_data_path))
    print("Length of EM data path: ", len(opt.EM_data_path))

    return opt


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        return_images = []
        
        for image in data:
            image = torch.unsqueeze(image.data, 0)
            if len(self.data) < self.max_size:
                # Buffer가 가득 차지 않은 경우 새 이미지 추가
                self.data.append(image)
                return_images.append(image)
            else:
                # Buffer가 가득 찬 경우, 일부는 기존 이미지, 일부는 새 이미지
                if random.uniform(0, 1) > 0.5:
                    # 기존 Buffer에서 샘플링
                    idx = random.randint(0, self.max_size - 1)
                    return_images.append(self.data[idx].clone())
                    self.data[idx] = image  # 대체
                else:
                    return_images.append(image)
        
        return torch.cat(return_images)
