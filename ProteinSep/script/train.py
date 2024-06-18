import os
import time
import random
import logging

import torch
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ProteinSep.model.ProteinSepNet import ProteinSepNet
from utils.ProteinSep.dataset import DataLoaderProteinSep
from utils.ProteinSep.util import parse_arguments
from utils.image_transform import random_transform

def train(dataloader, model, optimizer, rng, writer, epoch, opt):
    """ Train ProteinSepNet

    Args:
        dataloader (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        rng (_type_): _description_
        writer (_type_): _description_
        epoch (_type_): _description_
        opt (_type_): _description_
    """
    model.train()
    
    l1_loss_list = list()
    l2_loss_list = list()
    total_loss_list = list()
    
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    
    l1_l2_loss_coef = opt.l1_l2_loss_coef
    
    for idx, (data, _) in enumerate(tqdm(dataloader), desc="Train ProteinSepNet"):
        ##### Load data #####
        input_image, target_image_1, target_image_2 = random_transform(*data, rng)
        input_image = torch.unsqueeze(input_image, 1) # [N,C,Z,X,Y], C=1
        
        if not opt.use_CPU:
            input_image = input_image.cuda()
            target_image_1 = target_image_1.cuda()
            target_image_2 = target_image_2.cuda()
            
        target_image = torch.stack((target_image_1, target_image_2), dim=1) # [N,C,Z,X,Y], C=2
        
        ##### Update network #####
        optimizer.zero_grad()
        # ProtenSepNet
        separated_image = model(input_image)
        # l1/l2 loss
        l1_loss = L1_pixelwise(separated_image, target_image)
        l2_loss = L2_pixelwise(separated_image, target_image)
        # Total loss
        total_loss = l1_l2_loss_coef[0]*l1_loss + l1_l2_loss_coef[1]*l2_loss
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Logging purpose
        l1_loss_list.append(l1_loss.item())
        l2_loss_list.append(l2_loss.item())
        total_loss_list.append(total_loss.item())
        
        ##### Print log #####
        if (epoch % opt.logging_interval == 0) and (idx % opt.logging_interval_batch == 0):
            total_loss_mean = np.mean(np.array(total_loss_list))
            l1_loss_mean = np.mean(np.array(l1_loss_list))
            l2_loss_mean = np.mean(np.array(l2_loss_list))

            writer.add_scalar("Loss_total/train_batch", total_loss_mean, epoch*len(dataloader) + idx)
            writer.add_scalar("Loss_l1/train_batch", l1_loss_mean, epoch*len(dataloader) + idx)
            writer.add_scalar("Loss_l2/train_batch", l2_loss_mean, epoch*len(dataloader) + idx)
    
    total_loss_mean = np.mean(np.array(total_loss_list))
    l1_loss_mean = np.mean(np.array(l1_loss_list))
    l2_loss_mean = np.mean(np.array(l2_loss_list))
    
    return total_loss_mean, l1_loss_mean, l2_loss_mean

def evaluate(dataloader, model, opt):
    random_transform_rng = np.random.default_rng(opt.random_seed)
    with torch.no_grad():
        model.eval()
        
        l1_loss_list = list()
        l2_loss_list = list()
        total_loss_list = list()
        
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        
        l1_l2_loss_coef = opt.l1_l2_loss_coef
        
        for idx, (synthetic_data, real_data) in enumerate(tqdm(dataloader), desc="Evaluate ProteinSepNet"):
            data = synthetic_data
            ##### Load data #####
            input_image, target_image_1, target_image_2 = random_transform(*data, random_transform_rng)
            input_image = torch.unsqueeze(input_image, 1) # [N,C,Z,X,Y], C=1
            
            if not opt.use_CPU:
                input_image = input_image.cuda()
                target_image_1 = target_image_1.cuda()
                target_image_2 = target_image_2.cuda()
                
            target_image = torch.stack((target_image_1, target_image_2), dim=1) # [N,C,Z,X,Y], C=2
            
            ##### Update network #####
            # ProtenSepNet
            separated_image = model(input_image)
            # l1/l2 loss
            l1_loss = L1_pixelwise(separated_image, target_image)
            l2_loss = L2_pixelwise(separated_image, target_image)
            # Total loss
            total_loss = l1_l2_loss_coef[0]*l1_loss + l1_l2_loss_coef[1]*l2_loss
            
            # Logging purpose
            l1_loss_list.append(l1_loss.item())
            l2_loss_list.append(l2_loss.item())
            total_loss_list.append(total_loss.item())
        
    total_loss_mean = np.mean(np.array(total_loss_list))
    l1_loss_mean = np.mean(np.array(l1_loss_list))
    l2_loss_mean = np.mean(np.array(l2_loss_list))
            
    return total_loss_mean, l1_loss_mean, l2_loss_mean


if __name__=="__main__":
    random.seed(0)
    torch.manual_seed(0)
    
    ##### Initialize #####
    opt = parse_arguments()
    opt.use_CPU = True if not torch.cuda.is_available() else opt.use_CPU
    
    os.makedirs("{}/images/{}".format(opt.results_dir, opt.exp_name), exist_ok=True)
    os.makedirs("{}/saved_models/{}".format(opt.results_dir, opt.exp_name), exist_ok=True)
    os.makedirs("{}/logs".format(opt.document_dir), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename="{}/logs/{}.log".format(opt.document_dir, opt.exp_name),\
        filemode="a", format="%(name)s - %(levelname)s - %(message)s")
    writer = SummaryWriter("{}/tsboard/{}".format(opt.document_dir, opt.exp_name))

    opt.traindata_list = opt.traindata_list[:3]
    opt.testdata_list = [opt.testdata_list[-1]]
    
    ##### Dataset #####
    print("number of train dataset ({}): {}".format(", ".join(opt.protein_list), len(opt.traindata_list)))
    print("number of test dataset ({}): {}".format(", ".join(opt.protein_list), len(opt.testdata_list)))
    
    logging.info("number of train dataset ({}): {}".format(", ".join(opt.protein_list), len(opt.traindata_list)))
    logging.info("number of test dataset ({}): {}".format(", ".join(opt.protein_list), len(opt.testdata_list)))
    
    logging.info("TRAIN DATA")
    for data in opt.traindata_list:
        logging.info("      {}".format(data))
    
    logging.info("TEST DATA")
    for data in opt.testdata_list:
        logging.info("      {}".format(data))
        
    dataloader_train = DataLoaderProteinSep(opt.traindata_list, opt.protein_list, opt.patch_size, opt.batch_size, opt.batch_num, opt.random_seed, 
                                            opt.min_brightness, opt.brightness_range, opt.norm_type, opt.norm_quantile, shuffle=True)
    dataloader_test = DataLoaderProteinSep(opt.testdata_list, opt.protein_list, opt.patch_size, opt.batch_size, opt.batch_num, opt.random_seed, 
                                           opt.min_brightness, opt.brightness_range, opt.norm_type, opt.norm_quantile, shuffle=False)
    
    logging.info("")
    
    ##### Model, Optimizers, and Loss #####
    model = ProteinSepNet(nTarget=len(opt.protein_list), mid_channels=opt.mid_channels_ProteinSepNet)
    if not opt.use_CPU:
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    if opt.epoch != 0:
        model.load_state_dict(torch.load("{}/saved_models/{}/model_{}.pth".format(opt.results_dir, opt.exp_name, opt.epoch-1)))
        optimizer.load_state_dict(torch.load("{}/saved_models/{}/optimizer_{}.pth".format(opt.results_dir, opt.exp_name, opt.epoch-1)))
        print('Loaded pre-trained model and optimizer weights of epoch {}'.format(opt.epoch-1))
    
    ##### Train and Validation #####
    random_transform_rng = np.random.default_rng(opt.random_seed)
    for epoch in range(opt.epoch, opt.n_epochs):
        # train model
        model.train()
        total_loss, l1_loss, l2_loss = train(dataloader_train, model, optimizer, 
                                             random_transform_rng, writer, epoch, opt)
            
        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if (epoch % opt.logging_interval == 0):            
            writer.add_scalar("Loss_total/train", total_loss, epoch)
            writer.add_scalar("Loss_l1/train", l1_loss, epoch)
            writer.add_scalar("Loss_l2/train", l2_loss, epoch)
            
            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "+\
                f"loss : {total_loss:.4f}, loss_l1 : {l1_loss:.4f}, loss_l2 : {l2_loss:.4f}")

        # save model and optimizer
        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), "{}/saved_models/{}/model_{}.pth".format(opt.results_dir, opt.exp_name, epoch))
            torch.save(optimizer.state_dict(), "{}/saved_models/{}/optimizer_{}.pth".format(opt.results_dir, opt.exp_name, epoch))
            
        # evaluate model
        if (opt.test_interval != -1) and (epoch % opt.test_interval == 0):
            total_loss_mean, l1_loss_mean, l2_loss_mean = evaluate(dataloader_test, model, opt)
            writer.add_scalar("Loss_total/test", total_loss, epoch)
            writer.add_scalar("Loss_l1/test", l1_loss, epoch)
            writer.add_scalar("Loss_l2/test", l2_loss, epoch)