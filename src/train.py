import gc
import os
import random
import logging
import numpy as np
import torch
import itertools
import skimage.io as skio

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.discriminator_3d import Anisotrpic_MultiDiscriminator3D as Discriminator
from model.generator_3d_groupnorm import Anisotrpic_UNet3D as UNet
from src.utils.dataset import Light2EM_Dataset
from src.utils.util import parse_arguments, ReplayBuffer

import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision.utils as vutils


def compute_loss_D_adv(model, x, gt):
    """Computes the MSE between model output and scalar gt"""
    loss = 0
    # n=0
    output = model.forward(x)
    for out in output:
        squared_diff = (out - gt) ** 2
        loss += torch.mean(squared_diff)
        # n=n+1
    return loss


def run(rank, size, opt):
    torch.autograd.set_detect_anomaly(True)
    if size > 1:
        print(f"Rank {rank} is training")
    # ----------
    # Initialize
    # ----------
    random.seed(0)
    torch.manual_seed(0)
    if rank == 0:
        writer = SummaryWriter(opt.results_dir + "/tsboard/{}".format(opt.exp_name))

    #-----------
    # Dataset
    # ----------
    dataset_train = Light2EM_Dataset(opt.LM_data_path, opt.EM_data_path, opt.patch_size, opt.patch_overlap)
    if size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=size, rank=rank, shuffle=True)
        dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True, sampler=train_sampler)
    else:
        dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)
    # Calculate starting epoch and batch
    if opt.effective_iter != 0:
        start_epoch = opt.effective_iter // len(dataloader_train)
        start_batch = opt.effective_iter % len(dataloader_train)
    else:
        start_epoch = 0
        start_batch = 0
    
    print(f"Starting from epoch {start_epoch}, batch {start_batch}")

    # print length of dataloader
    print(f"Rank {rank} has {len(dataloader_train)} batches")
    
    if opt.replay_buffer_size > 0:
        replay_buffer_A = ReplayBuffer(max_size=opt.replay_buffer_size)
        replay_buffer_B = ReplayBuffer(max_size=opt.replay_buffer_size)

    # ----------
    # Model, Optimizers, and Loss
    # ----------
    G_AB = UNet(n_channels=1, n_classes=1, num_channels=opt.generator_channels).to(rank)
    G_BA = UNet(n_channels=1, n_classes=1, num_channels=opt.generator_channels).to(rank)
    D_A = Discriminator(channels=1, num_scales=opt.discriminator_num_scales, num_filters=opt.discriminator_channels).to(rank)
    D_B = Discriminator(channels=1, num_scales=opt.discriminator_num_scales, num_filters=opt.discriminator_channels).to(rank)
    
    if size > 1:
        G_AB = torch.nn.parallel.DistributedDataParallel(G_AB, device_ids=[rank])
        G_BA = torch.nn.parallel.DistributedDataParallel(G_BA, device_ids=[rank])
        D_A = torch.nn.parallel.DistributedDataParallel(D_A, device_ids=[rank])
        D_B = torch.nn.parallel.DistributedDataParallel(D_B, device_ids=[rank])

    opt.lr = opt.lr * opt.batch_size # Scale the learning rate by the batch size
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Add schedulers
    if opt.use_scheduler:
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_G, 
            T_0=10000,
            T_mult=2,  # Double the restart interval after each restart
            eta_min=opt.lr * 0.01  # Minimum learning rate
        )
        scheduler_D_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_D_A,
            T_0=10000,
            T_mult=2,
            eta_min=opt.lr * 0.01
        )
        scheduler_D_B = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_D_B,
            T_0=10000,
            T_mult=2,
            eta_min=opt.lr * 0.01
        )

    if opt.effective_iter != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/G_AB_{}.pth".format(opt.exp_name, opt.effective_iter)))
        G_BA.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/G_BA_{}.pth".format(opt.exp_name, opt.effective_iter)))
        D_A.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/D_A_{}.pth".format(opt.exp_name, opt.effective_iter)))
        D_B.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/D_B_{}.pth".format(opt.exp_name, opt.effective_iter)))
        optimizer_G.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/optimizer_G_{}.pth".format(opt.exp_name, opt.effective_iter)))
        optimizer_D_A.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/optimizer_D_A_{}.pth".format(opt.exp_name, opt.effective_iter)))
        optimizer_D_B.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/optimizer_D_B_{}.pth".format(opt.exp_name, opt.effective_iter)))
        # Load scheduler states
        if opt.use_scheduler:
            scheduler_G.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/scheduler_G_{}.pth".format(opt.exp_name, opt.effective_iter)))
            scheduler_D_A.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/scheduler_D_A_{}.pth".format(opt.exp_name, opt.effective_iter)))
            scheduler_D_B.load_state_dict(torch.load(opt.results_dir + "/saved_models/{}/scheduler_D_B_{}.pth".format(opt.exp_name, opt.effective_iter)))
        print("Loaded pretrained models")
    
    L2_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    # ----------
    # Training
    # ----------
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    for epoch in range(start_epoch, opt.n_epochs):
        for i, batch in enumerate(tqdm(dataloader_train)):
            real_A = batch["A"].to(rank)
            real_B = batch["B"].to(rank)
            real_A = real_A / 255.0
            real_B = real_B / 255.0

            # ----------
            # Generators
            # ----------
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = compute_loss_D_adv(D_B, fake_B, 1)

            fake_A = G_BA(real_B)
            loss_GAN_BA = compute_loss_D_adv(D_A, fake_A, 1)

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = L1_loss(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = L1_loss(recov_B, real_B)

            # Group consistency loss            
            real_A_group1 = torch.sigmoid((real_A - (110 / 255.0))*100)
            real_A_group2 = torch.sigmoid(((110 / 255.0) - real_A)*100)
            fake_A_group1 = torch.sigmoid((fake_A - (110 / 255.0))*100)
            fake_A_group2 = torch.sigmoid(((110 / 255.0) - fake_A)*100)
            real_B_group1 = torch.sigmoid((real_B - (120 / 255.0))*100) - torch.sigmoid((real_B - (160 / 255.0))*100)
            real_B_group2 = torch.sigmoid(((120 / 255.0) - real_B)*100)
            real_B_group3 = torch.sigmoid((real_B - (160 / 255.0))*100)
            fake_B_group1 = torch.sigmoid((fake_B - (120 / 255.0))*100) - torch.sigmoid((fake_B - (160 / 255.0))*100)
            fake_B_group2 = torch.sigmoid(((120 / 255.0) - fake_B)*100)
            fake_B_group3 = torch.sigmoid((fake_B - (160 / 255.0))*100)

            loss_content_A_group1 = L1_loss(real_A_group1, fake_B_group1*real_A_group1)
            loss_content_A_group2 = L1_loss(real_A_group2, (fake_B_group2+fake_B_group3)*real_A_group2)
            loss_content_B_group1 = L1_loss(real_B_group1, fake_A_group1*real_B_group1)
            loss_content_B_group2 = L1_loss(real_B_group2, fake_A_group2*real_B_group2)
            loss_content_B_group3 = L1_loss(real_B_group3, fake_A_group2*real_B_group3)

            content_loss_rate_exponential = 2 ** (-(epoch * len(dataloader_train) + i) / opt.content_loss_decay_batch)
            loss_content = 2 / 5 * (loss_content_A_group1 + loss_content_A_group2 + loss_content_B_group1 + loss_content_B_group2 + loss_content_B_group3) * content_loss_rate_exponential

            # Total loss
            loss_G = loss_GAN_AB + loss_GAN_BA + opt.lambda_cyc * (loss_cycle_A + loss_cycle_B) + opt.lambda_content * loss_content
            loss_G.backward()
            if opt.gradient_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(G_AB.parameters(), opt.gradient_max_norm)
                torch.nn.utils.clip_grad_norm_(G_BA.parameters(), opt.gradient_max_norm)
            optimizer_G.step()

            # ----------
            # Discriminator A
            # ----------
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = compute_loss_D_adv(D_A, real_A, 1)
            # Fake loss
            if opt.replay_buffer_size > 0:
                fake_A_replay = replay_buffer_A.push_and_pop(fake_A)
                loss_fake = compute_loss_D_adv(D_A, fake_A_replay.detach(), 0)
            else:
                loss_fake = compute_loss_D_adv(D_A, fake_A.detach(), 0)
            # Total loss
            loss_D_A = 0.5 * (loss_real + loss_fake)
            loss_D_A.backward()
            if opt.gradient_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(D_A.parameters(), opt.gradient_max_norm)
            optimizer_D_A.step()

            # ----------
            # Discriminator B
            # ----------
            optimizer_D_B.zero_grad()
            # Real loss
            loss_real = compute_loss_D_adv(D_B, real_B, 1)
            # Fake loss
            if opt.replay_buffer_size > 0:
                fake_B_replay = replay_buffer_B.push_and_pop(fake_B)
                loss_fake = compute_loss_D_adv(D_B, fake_B_replay.detach(), 0)
            else:
                loss_fake = compute_loss_D_adv(D_B, fake_B.detach(), 0)
            # Total loss
            loss_D_B = 0.5 * (loss_real + loss_fake)
            loss_D_B.backward()
            if opt.gradient_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(D_B.parameters(), opt.gradient_max_norm)
            optimizer_D_B.step()

            if opt.use_scheduler:
                scheduler_G.step()
                scheduler_D_A.step()
                scheduler_D_B.step()

            # ----------
            # Logging
            # ----------
            if rank == 0:
                if opt.effective_iter == 0:
                    effective_iter = epoch * len(dataloader_train) + i
                else:
                    effective_iter = opt.effective_iter + 1 + epoch * len(dataloader_train) + i
                if effective_iter % opt.logging_interval_batch == 0:
                    logging.info("[Epoch {}/{}] [Batch {}/{}] [D_A loss: {}] [D_B loss: {}] [G loss: {}]".format(epoch, opt.n_epochs, i, len(dataloader_train), loss_D_A.item(), loss_D_B.item(), loss_G.item()))
                    writer.add_scalar("D_A loss", loss_D_A.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("D_B loss", loss_D_B.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("loss_GAN_AB", opt.lambda_gen * loss_GAN_AB.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("loss_GAN_BA", opt.lambda_gen * loss_GAN_BA.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("G loss", loss_G.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss", opt.lambda_content * loss_content.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss A group 1", loss_content_A_group1.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss A group 2", loss_content_A_group2.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss B group 1", loss_content_B_group1.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss B group 2", loss_content_B_group2.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Content loss B group 3", loss_content_B_group3.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Cycle loss A", opt.lambda_cyc * loss_cycle_A.item(), epoch * len(dataloader_train) + i)
                    writer.add_scalar("Cycle loss B", opt.lambda_cyc * loss_cycle_B.item(), epoch * len(dataloader_train) + i)
                    # Learning rate logging
                    current_lr_G = optimizer_G.param_groups[0]['lr']
                    current_lr_D_A = optimizer_D_A.param_groups[0]['lr']
                    current_lr_D_B = optimizer_D_B.param_groups[0]['lr']
                    
                    writer.add_scalar("Learning_Rate/Generator", current_lr_G, effective_iter)
                    writer.add_scalar("Learning_Rate/Discriminator_A", current_lr_D_A, effective_iter)
                    writer.add_scalar("Learning_Rate/Discriminator_B", current_lr_D_B, effective_iter)
                    
                    # Log to console as well
                    logging.info(f"Learning rates - G: {current_lr_G:.6f}, D_A: {current_lr_D_A:.6f}, D_B: {current_lr_D_B:.6f}")
                
                if effective_iter % opt.image_save_interval_batch == 0:
                    # Save images
                    real_A_numpy = real_A[0].cpu().detach()
                    real_B_numpy = real_B[0].cpu().detach()
                    fake_A_numpy = fake_A[0].cpu().detach()
                    fake_B_numpy = fake_B[0].cpu().detach()
                    recov_A_numpy = recov_A[0].cpu().detach()
                    recov_B_numpy = recov_B[0].cpu().detach()
                    skio.imsave(opt.results_dir + "/images/{}/{}_real_A.tif".format(opt.exp_name, effective_iter), real_A_numpy.numpy())
                    skio.imsave(opt.results_dir + "/images/{}/{}_real_B.tif".format(opt.exp_name, effective_iter), real_B_numpy.numpy())
                    skio.imsave(opt.results_dir + "/images/{}/{}_fake_A.tif".format(opt.exp_name, effective_iter), fake_A_numpy.numpy())
                    skio.imsave(opt.results_dir + "/images/{}/{}_fake_B.tif".format(opt.exp_name, effective_iter), fake_B_numpy.numpy())
                    skio.imsave(opt.results_dir + "/images/{}/{}_recov_A.tif".format(opt.exp_name, effective_iter), recov_A_numpy.numpy())
                    skio.imsave(opt.results_dir + "/images/{}/{}_recov_B.tif".format(opt.exp_name, effective_iter), recov_B_numpy.numpy())
                    grid = vutils.make_grid([real_A_numpy[0, 8:9], fake_B_numpy[0, 8:9], recov_A_numpy[0, 8:9]], normalize=True, scale_each=True)
                    writer.add_image("A2B/batch0", grid, effective_iter)
                    grid = vutils.make_grid([real_B_numpy[0, 8:9], fake_A_numpy[0, 8:9], recov_B_numpy[0, 8:9]], normalize=True, scale_each=True)
                    writer.add_image("B2A/batch0", grid, effective_iter)
                    
                
                if effective_iter % opt.checkpoint_interval_batch == 0:
                    torch.save(G_AB.state_dict(), opt.results_dir + "/saved_models/{}/G_AB_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(G_BA.state_dict(), opt.results_dir + "/saved_models/{}/G_BA_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(D_A.state_dict(), opt.results_dir + "/saved_models/{}/D_A_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(D_B.state_dict(), opt.results_dir + "/saved_models/{}/D_B_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(optimizer_G.state_dict(), opt.results_dir + "/saved_models/{}/optimizer_G_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(optimizer_D_A.state_dict(), opt.results_dir + "/saved_models/{}/optimizer_D_A_{}.pth".format(opt.exp_name, effective_iter))
                    torch.save(optimizer_D_B.state_dict(), opt.results_dir + "/saved_models/{}/optimizer_D_B_{}.pth".format(opt.exp_name, effective_iter))
                    # Save scheduler states
                    if opt.use_scheduler:
                        torch.save(scheduler_G.state_dict(), opt.results_dir + "/saved_models/{}/scheduler_G_{}.pth".format(opt.exp_name, effective_iter))
                        torch.save(scheduler_D_A.state_dict(), opt.results_dir + "/saved_models/{}/scheduler_D_A_{}.pth".format(opt.exp_name, effective_iter))
                        torch.save(scheduler_D_B.state_dict(), opt.results_dir + "/saved_models/{}/scheduler_D_B_{}.pth".format(opt.exp_name, effective_iter))


def init_process(rank, size, fn, opt, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, opt)


if __name__=="__main__":
    torch.cuda.empty_cache()
    # ----------
    # Initialize: Create sample and checkpoint directories
    # ----------
    opt = parse_arguments()
    cuda = torch.cuda.is_available()
    if not cuda:
        raise Exception("CUDA required!")

    os.makedirs(opt.results_dir + "/images/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/saved_models/{}".format(opt.exp_name), exist_ok=True)
    os.makedirs(opt.results_dir + "/logs".format(opt.exp_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=opt.results_dir + "/logs/{}.log".format(opt.exp_name),\
        filemode="a", format="%(name)s - %(levelname)s - %(message)s")
    
    # ----------
    # Set up multiple processes
    # ----------
    size = 1
    if size > 1:
        processes = []
        mp.set_start_method("spawn", force=True)
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, run, opt))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    else:
        run(0, 1, opt)
    