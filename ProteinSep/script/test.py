import os
import yaml
import argparse
import tifffile

import torch
import numpy as np
import skimage.io as skio

from argparse import Namespace

from ProteinSep.model.ProteinSepNet import ProteinSepNet
from utils.image_normalize import return_norm_func

def replace2p(x):
    x = str(x).replace('.', 'p')
    if len(x) == 1:
        x += "p0"
    return x


if __name__=="__main__":
    ##### Initialize #####
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace_file", type=str, help="path to the namespace file")
    parser.add_argument("--test_epoch", type=int, default=0, help="epoch to test the model")
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    test_opt = parser.parse_args()
    
    with open(test_opt.namespace_file) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = Namespace(**opt)
    print(opt.exp_name)
    
    ##### Load model #####
    model = ProteinSepNet(nTarget=len(opt.protein_list), mid_channels=opt.mid_channels_ProteinSepNet)
    model.load_state_dict(torch.load("{}/saved_models/{}/model_{}.pth".\
        format(opt.results_dir, opt.exp_name, test_opt.test_epoch)))
    
    if not opt.use_CPU:
        model = model.cuda()
    print('Loaded pre-trained model of epoch {}'.format(test_opt.test_epoch))
    
    norm_func = return_norm_func(opt.norm_type)

    #### Make results folder #####
    results_dir = "{}/images/{}/{}/".format(opt.results_dir, "/".join(opt.data_dir.split("/")[-2:]), opt.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    ##### Inference #####
    for image_file in opt.testdata_list:
        # Load data
        image_c1 = torch.from_numpy(skio.imread("{}_ch1.tif".format(image_file)).astype(np.float32))
        image_c2 = torch.from_numpy(skio.imread("{}_ch2.tif".format(image_file)).astype(np.float32))
        image_real = torch.from_numpy(skio.imread("{}_ch3.tif".format(image_file)).astype(np.float32))
        
        image_c1, image_c1_wobg = norm_func(image_c1, *opt.norm_quantile)
        image_c2, image_c2_wobg = norm_func(image_c2, *opt.norm_quantile)
        image_real, _ = norm_func(image_real, *opt.norm_quantile)
            
        image_target = torch.stack([image_c1_wobg, image_c2_wobg])
        
        print("Loaded", image_file)
        
        # real mixed image inference
        image_real = torch.unsqueeze(torch.unsqueeze(image_real, 0), 0)
        if not opt.use_CPU:
            image_real = image_real.cuda()
            
        model.eval()
        with torch.no_grad():
            try:
                separated_real = model(image_real[:,:,:16]).cpu().squeeze()
                for idx in range(image_real.shape[2]-separated_real.shape[1]):
                    separated_real_ = model(image_real[:,:,idx+1:idx+17]).cpu().squeeze()
                    separated_real = torch.concat([separated_real, separated_real_[:,-1:]], axis=1)
            except:
                breakpoint()
                separated_real = model(image_real[:,:-1,:,:]).cpu().squeeze()

        tifffile.imwrite("{}/target.tif".format(results_dir), image_target.numpy())
        tifffile.imwrite("{}/real_input.tif".format(results_dir, test_opt.test_epoch), image_real.cpu().numpy())
        tifffile.imwrite("{}/real_separated_epoch_{}.tif".format(results_dir, test_opt.test_epoch), separated_real.numpy())
        
        # synthetic mixed image inference
        for a in range(11):
            image_synthetic = (a/10) * image_c1 + (1-a/10) * image_c2
            image_target = torch.stack([image_c1_wobg, image_c2_wobg])
            
            image_synthetic = torch.unsqueeze(torch.unsqueeze(image_synthetic, 0), 0)
            if not opt.use_CPU:
                image_synthetic = image_synthetic.cuda()
                
            model.eval()
            with torch.no_grad():
                try:
                    separated_synthetic = model(image_synthetic[:,:,:16]).cpu().squeeze()
                    for idx in range(image_synthetic.shape[2]-separated_synthetic.shape[1]):
                        separated_synthetic_ = model(image_synthetic[:,:,idx+1:idx+17]).cpu().squeeze()
                        separated_synthetic = torch.concat([separated_synthetic, separated_synthetic_[:,-1:]], axis=1)
                except:
                    breakpoint()
                    separated_synthetic = model(image_synthetic[:,:-1,:,:].cuda()).cpu().squeeze()

            tifffile.imwrite("{}/synthetic_input_{}.tif".format(results_dir,  replace2p(a/10)), image_synthetic.cpu().numpy())
            tifffile.imwrite("{}/synthetic_separated_{}_epoch_{}.tif".format(results_dir, replace2p(a/10), test_opt.test_epoch), separated_synthetic.numpy())
