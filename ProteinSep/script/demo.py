import os
import glob
import yaml
import argparse
import tifffile

import torch
import numpy as np
import skimage.io as skio

from argparse import Namespace

from ProteinSep.model.ProteinSepNet_v3 import ProteinSepNet
# from ProteinSep.model.Discriminator import Discriminator
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
    parser.add_argument("--testdata_folder", type=str, default="", help="path to the test data folder")
    parser.add_argument("--test_epoch", type=int, default=0, help="epoch to test the model")
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    test_opt = parser.parse_args()
    
    with open(test_opt.namespace_file) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = Namespace(**opt)
    print(opt.exp_name)
    
    if test_opt.testdata_folder != "":
        test_opt.testdata_list = sorted(glob.glob(test_opt.testdata_folder + "/*.tif"))
    else:
        raise Exception("The testdata directory needed")
    
    ##### Load model #####
    model = ProteinSepNet(nTarget=len(opt.protein_list), mid_channels=opt.mid_channels_ProteinSepNet)
    model.load_state_dict(torch.load("{}/saved_models/{}/model_{}.pth".\
        format(opt.results_dir, opt.exp_name, test_opt.test_epoch)))
    
    if not opt.use_CPU:
        model = model.cuda()
    print('Loaded pre-trained model of epoch {}'.format(test_opt.test_epoch))
    
    norm_func = return_norm_func(opt.norm_type)
    
    #### Make results folder #####
    results_dir = "{}/images/{}/{}".format(opt.results_dir, "/".join(test_opt.testdata_dir.split("/")[-2:]), opt.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    ##### Inference #####
    for image_file in test_opt.testdata_list:
        # Load data
        image_real = torch.from_numpy(skio.imread(image_file).astype(np.float32))
        image_real, _ = norm_func(image_real, *opt.norm_quantile)
            
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

        tifffile.imwrite("{}/{}_input.tif".format(results_dir, image_file.split("/")[-1][:-4]), image_real.cpu().numpy())
        tifffile.imwrite("{}/{}_separated_epoch_{}.tif".format(results_dir, image_file.split("/")[-1][:-4], test_opt.test_epoch), separated_real.numpy())
