import os
import time
import random
import logging

import torch
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from FeatureExtract.model.FeatureExtractNet import FeatureExtractNet
from utils.FeatureExtract.dimension_reduction import calc_tsne, twodim_visualization
from utils.FeatureExtract.dataset import DataLoaderFeatureExtract
from utils.FeatureExtract.util import parse_arguments
from utils.image_transform import random_transform

def train(dataloader, model, optimizer, rng, writer, epoch, opt):
    """ Train FeatureExtractNet
    
    Args:
        dataloader (torch DataLoader): _description_
        model (torch nn.Module): _description_
        optimizer (torch optimizer): _description_
        rng (np.random.default_rng): numpy random number generator
        writer (tensorboard SummaryWriter): _description_
        epoch (int): _description_
        opt (argparse.Namespace): _description_
    """
    model.train()
    
    loss_list_inter = list()
    loss_list_intra = list()
    loss_list_total = list()
    
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    
    l1_l2_loss_coef = opt.l1_l2_loss_coef
    inter_intra_loss_coef = opt.inter_intra_loss_coef
    inter_margin = opt.inter_margin
    intra_margin = opt.intra_margin
    
    for idx, data in enumerate(tqdm(dataloader, desc="Evaluate ProteinSepNet")):
        ##### Load data #####
        image, _, image_label = data
        image_label_list = list(set(image_label.tolist()))
        
        if not opt.use_CPU:
            image_input = image.cuda()
        
        image_input, _, _ = random_transform(image_input, None, None, rng)
        
        ##### Update FeatureExtractNet #####
        optimizer.zero_grad()
        latent_feature = model(image_input)
        
        # inter-class loss
        mean_features = list()
        loss_inter_total = list()
        
        for label in image_label_list:
            mean_feature = torch.mean(latent_feature[image_label==label], axis=0)
            mean_features.append(mean_feature)
            
            loss_inter_total_tmp = list()
            for label_idx in range(len(image_label)):
                if image_label[label_idx] == label:
                    loss_l1 = L1_pixelwise(mean_feature, latent_feature[label_idx])
                    loss_l2 = L2_pixelwise(mean_feature, latent_feature[label_idx])
                    
                    loss_l1_l2 = l1_l2_loss_coef[0]*loss_l1 + l1_l2_loss_coef[1]*loss_l2
                    loss_inter = torch.clamp(loss_l1_l2 - inter_margin, min=0)
                    
                    loss_inter_total_tmp.append(loss_inter)
            loss_inter_total.append(torch.mean(torch.stack(loss_inter_total_tmp)))
        loss_inter_total = torch.mean(torch.stack(loss_inter_total))
        
        # intra-class loss
        loss_intra_total = list()
        for label_idx1 in range(len(image_label_list)):
            for label_idx2 in range(label_idx1+1, len(image_label_list)):
                loss_l1 = L1_pixelwise(mean_features[label_idx1], mean_features[label_idx2])
                loss_l2 = L2_pixelwise(mean_features[label_idx1], mean_features[label_idx2])
                
                loss_l1_l2 = l1_l2_loss_coef[0]*loss_l1 + l1_l2_loss_coef[1]*loss_l2
                loss_intra = torch.clamp(2*intra_margin - loss_l1_l2, min=0)
                
                loss_intra_total.append(loss_intra)
        loss_intra_total = torch.mean(torch.stack(loss_intra_total))
        
        # 
        loss_total = inter_intra_loss_coef[0]*loss_inter_total \
            + inter_intra_loss_coef[1]*loss_intra_total
        
        # network backprop
        loss_total.backward()
        optimizer.step()
        
        ##### Append to list #####
        loss_list_inter.append(loss_inter_total.item())
        loss_list_intra.append(loss_intra_total.item())
        loss_list_total.append(loss_total.item())
        
        ##### Print log #####
        if (epoch % opt.logging_interval == 0) and (idx % opt.logging_interval_batch == 0):
            loss_mean_total = np.mean(np.array(loss_list_total))
            loss_mean_inter = np.mean(np.array(loss_list_inter))
            loss_mean_intra = np.mean(np.array(loss_list_intra))
            
            writer.add_scalar("Loss_total/train_batch", loss_mean_total, epoch*len(dataloader) + idx)
            writer.add_scalar("Loss_inter/train_batch", loss_mean_inter, epoch*len(dataloader) + idx)
            writer.add_scalar("Loss_intra/train_batch", loss_mean_intra, epoch*len(dataloader) + idx)
    
    loss_mean_total = np.mean(np.array(loss_list_total))
    loss_mean_inter = np.mean(np.array(loss_list_inter))
    loss_mean_intra = np.mean(np.array(loss_list_intra))
    
    return loss_mean_total, loss_mean_inter, loss_mean_intra

def evaluate(dataloader, model, opt):
    """ Evaluate FeatureExtractNet
    
    Args:
        dataloader (torch DataLoader): _description_
        model (torch nn.Module): _description_
        opt (argparse.Namespace): _description_
    """
    images = list()
    image_labels = list()
    latent_features = list()
    
    random_transform_rng = np.random.default_rng(opt.random_seed)
    with torch.no_grad():
        model.eval()
        
        loss_list_inter = list()
        loss_list_intra = list()
        loss_list_total = list()
        
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        
        l1_l2_loss_coef = opt.l1_l2_loss_coef
        inter_intra_loss_coef = opt.inter_intra_loss_coef
        inter_margin = opt.inter_margin
        intra_margin = opt.intra_margin
        
        for idx, data in enumerate(tqdm(dataloader, desc="evaluation")):
            ##### Load data #####
            image, _, image_label = data
            image_label_list = list(set(image_label.tolist()))
            
            if not opt.use_CPU:
                input_image = image.cuda()
            
            input_image, _, _ = random_transform(input_image, None, None, random_transform_rng)
            
            ##### Inference #####
            latent_feature = model(input_image)
            
            ##### Calculate loss #####
            # inter-class loss
            mean_features = list()
            loss_inter_total = list()
            
            for label in image_label_list:
                mean_feature = torch.mean(latent_feature[image_label==label], axis=0)
                mean_features.append(mean_feature)
                
                loss_inter_total_tmp = list()
                for label_idx in range(len(image_label)):
                    if image_label[label_idx] == label:
                        loss_l1 = L1_pixelwise(mean_feature, latent_feature[label_idx])
                        loss_l2 = L2_pixelwise(mean_feature, latent_feature[label_idx])
                        
                        loss_l1_l2 = l1_l2_loss_coef[0]*loss_l1 + l1_l2_loss_coef[1]*loss_l2
                        loss_inter = torch.clamp(loss_l1_l2 - inter_margin, min=0)
                        
                        loss_inter_total_tmp.append(loss_inter)
                loss_inter_total.append(torch.mean(torch.stack(loss_inter_total_tmp)))
            loss_inter_total = torch.mean(torch.stack(loss_inter_total))
            
            # intra-class loss
            loss_intra_total = list()
            for label_idx1 in range(len(image_label_list)):
                for label_idx2 in range(label_idx1+1, len(image_label_list)):
                    loss_l1 = L1_pixelwise(mean_features[label_idx1], mean_features[label_idx2])
                    loss_l2 = L2_pixelwise(mean_features[label_idx1], mean_features[label_idx2])
                    
                    loss_l1_l2 = l1_l2_loss_coef[0]*loss_l1 + l1_l2_loss_coef[1]*loss_l2
                    loss_intra = torch.clamp(2*intra_margin - loss_l1_l2, min=0)
                    
                    loss_intra_total.append(loss_intra)
            loss_intra_total = torch.mean(torch.stack(loss_intra_total))
            
            # 
            loss_total = inter_intra_loss_coef[0]*loss_inter_total \
                + inter_intra_loss_coef[1]*loss_intra_total
            
            ##### Append to list #####
            loss_list_inter.append(loss_inter_total.item())
            loss_list_intra.append(loss_intra_total.item())
            loss_list_total.append(loss_total.item())
            
            images.append(image.numpy())
            image_labels.append(image_label.numpy())
            latent_features.append(latent_feature.cpu().numpy())
    
    loss_mean_total = np.mean(np.array(loss_list_total))
    loss_mean_inter = np.mean(np.array(loss_list_inter))
    loss_mean_intra = np.mean(np.array(loss_list_intra))
    
    images = np.concatenate(images)
    image_labels = np.concatenate(image_labels)
    latent_features = np.concatenate(latent_features)
    
    return images, image_labels, latent_features, loss_mean_total, loss_mean_inter, loss_mean_intra


if __name__=="__main__":
    random.seed(0)
    torch.manual_seed(0)
    
    ##### Initialize #####
    opt = parse_arguments()
    opt.use_CPU = True if not torch.cuda.is_available() else opt.use_CPU
    
    os.makedirs("{}/images/{}".format(opt.results_dir, opt.exp_name), exist_ok=True)
    os.makedirs("{}/saved_models/{}".format(opt.results_dir, opt.exp_name), exist_ok=True)
    os.makedirs("{}/logs".format(opt.document_dir), exist_ok=True)
    os.makedirs("{}/tsne/{}".format(opt.document_dir, opt.exp_name), exist_ok=True)
    os.makedirs("{}/feature-based_distance/{}".format(opt.document_dir, opt.exp_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename="{}/logs/{}.log".format(opt.document_dir, opt.exp_name),\
        filemode="a", format="%(name)s - %(levelname)s - %(message)s")
    writer = SummaryWriter("{}/tsboard/{}".format(opt.document_dir, opt.exp_name))
    
    ##### Dataset #####
    logging.info("number of train dataset: {}".format(len(opt.traindata_list)))
    logging.info("number of train dataset: {}".format(len(opt.testdata_list)))
    
    logging.info("TRAIN DATA")
    for data in opt.traindata_list:
        logging.info("      {}".format(data))
    
    logging.info("TEST DATA")
    for data in opt.testdata_list:
        logging.info("      {}".format(data))
    
    logging.info("")
    
    print("number of train dataset: {}".format(len(opt.traindata_list)))
    dataloader_train = DataLoaderFeatureExtract(opt.traindata_list, opt.protein_list, opt.patch_size, opt.batch_size, opt.batch_num, 
                                          opt.random_seed, opt.brightness_range, opt.norm_type, opt.norm_quantile, shuffle=True)
    print("number of test dataset: {}".format(len(opt.testdata_list)))
    dataloader_test = DataLoaderFeatureExtract(opt.testdata_list, opt.protein_list, opt.patch_size, 256, 20, 
                                         opt.random_seed, opt.brightness_range, opt.norm_type, opt.norm_quantile, shuffle=True)
    
    ##### Model, Optimizers, and Loss #####
    model = FeatureExtractNet(opt.patch_size[-1], nColor=1, mid_channels=opt.mid_channels_FeatureExtractNet, 
                              out_features=opt.out_features_FeatureExtractNet)
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
        loss_total, loss_inter, loss_intra = train(dataloader_train, model, optimizer,
                                                   random_transform_rng, writer, epoch, opt)
        
        # logging
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if (epoch % opt.logging_interval == 0):            
            writer.add_scalar("Loss_total/train", loss_total, epoch)
            writer.add_scalar("Loss_inter/train", loss_inter, epoch)
            writer.add_scalar("Loss_intra/train", loss_intra, epoch)
            
            logging.info(f"[{ts}] Epoch [{epoch}/{opt.n_epochs}] "+\
                f"loss : {loss_total:.4f}, loss_inter : {loss_inter:.4f}, loss_intra : {loss_intra:.4f}")
        
        # save model and optimizer
        if (opt.checkpoint_interval != -1) and (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), "{}/saved_models/{}/model_{}.pth".format(opt.results_dir, opt.exp_name, epoch))
            torch.save(optimizer.state_dict(), "{}/saved_models/{}/optimizer_{}.pth".format(opt.results_dir, opt.exp_name, epoch))
        
        # evaluate model
        if (opt.test_interval != -1) and (epoch % opt.test_interval == 0):
            images, image_labels, latent_features, loss_mean_total, loss_mean_inter, loss_mean_intra \
                = evaluate(dataloader_test, model, opt)
            writer.add_scalar("Loss_total/test", loss_total, epoch)
            writer.add_scalar("Loss_inter/test", loss_inter, epoch)
            writer.add_scalar("Loss_intra/test", loss_intra, epoch)
            
            tsne_features = calc_tsne(latent_features)
            twodim_visualization(tsne_features, image_labels, opt, epoch, "tsne")
