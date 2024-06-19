import os
import glob
import yaml
import argparse
import warnings

from itertools import permutations
from utils.load_namespace import load_namespace

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    # Experiment profile
    parser.add_argument("--exp_name", type=str, default="mytest_ProteinSepNet", help="name of the experiment")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for rng")
    parser.add_argument("--results_dir", type=str, default="./results/ProteinSep", help="root directory to save results")
    parser.add_argument("--document_dir", type=str, default="./results/ProteinSep", help="root directory to save log and namesapce")
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default=["./data/sample_data"], nargs="+", help="root directory to the data")
    parser.add_argument("--protein_list", type=str, default=['CALB2', 'Calnexin'], nargs="+", help="list of the protein for separation")
    # Image normalization
    parser.add_argument("--norm_type", type=str, default="zstack", help="type of normalization")
    parser.add_argument("--norm_quantile", type=float, default=[0.10, 0.99], nargs="+", help="lower and upper quantile used for normalization")
    
    # Train
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (need epoch-1 model)")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--patch_size", type=int, default=[16,512,512], nargs="+", help="size of the patches")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--batch_num", type=int, default=1000, help="the number of batches for each epoch")
    parser.add_argument("--min_brightness", type=float, default=0.01, help="minimum of alpha or beta")
    parser.add_argument("--brightness_range", type=float, default=[0.8,1.2], nargs="+", help="range of alpha + beta")
    
    parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")
    parser.add_argument("--l1_l2_loss_coef", type=float, default=[0.5,0.5], nargs="+", help="L1/L2 loss coefficients")
    
    parser.add_argument("--mid_channels_ProteinSepNet", type=int, default=[8,12,16,32,48,64], nargs="+", help="the number of channels of ColorSepNet")
    parser.add_argument("--one_by_one_conv_channels_ProteinSepNet", type=int, default=[6,4], nargs="+", help="the number of 1x1 channels of ColorSepNet")

    # Util
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--logging_interval_batch", type=int, default=50, help="interval between logging info (in batches)")
    parser.add_argument("--logging_interval", type=int, default=1, help="interval between logging info (in epochs)")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="interval between model checkpoints (in epochs)")
    parser.add_argument("--test_interval", type=int, default=25, help="interval between test checkpoints (in epochs)")
    
    opt = parser.parse_args()
    
    # check parameters
    if len(opt.protein_list) != 2:
        raise Exception("not yet implemented")
    if len(opt.brightness_range) != 2:
        raise Exception("brightness range must be composite of two values, minimum brightness and maximum brightness")
    
    opt.protein_list = sorted(opt.protein_list)
    opt.exp_name = "{}_{}".format(opt.exp_name, "_".join(opt.protein_list))
    
    # Namespace directory
    os.makedirs("{}/namespace".format(opt.document_dir), exist_ok=True)
    namespace_file = "{}/namespace/{}.yaml".format(opt.document_dir, opt.exp_name)
    
    try:
        # Load Namespace
        opt = load_namespace(namespace_file, opt)

    except:
        # Split train and test data
        opt.traindata_list = list()
        opt.testdata_list = list()
        for folder in opt.data_dir:
            # image file organization
            # : *{protein α}_{protein β}_{sample idx}_ch1.tif    individual image of protein α
            # : *{protein α}_{protein β}_{sample idx}_ch2.tif    individual image of protein β
            # : *{protein α}_{protein β}_{sample idx}_ch3.tif    mixed image of protein α and protein β
            
            opt.traindata_list += glob.glob("{}/train/*.tif".format(folder))
            opt.testdata_list += glob.glob("{}/test/*.tif".format(folder))
        
        # remove suffix "_ch*.tif"
        opt.traindata_list = set([data[:-8] for data in opt.traindata_list])
        opt.testdata_list = set([data[:-8] for data in opt.testdata_list])
        
        opt.traindata_list = sorted(list(opt.traindata_list))
        opt.testdata_list = sorted(list(opt.testdata_list))
        
        if len(opt.traindata_list) == 0:
            raise Exception("There is no train data")
        if len(opt.testdata_list) == 0:
            warnings.warn("There is no test data")
        
        # Save Namespace
        with open(namespace_file, 'w') as f:
            yaml.dump(vars(opt), f)
            
    return opt
            
    
if __name__=="__main__":
    parse_arguments()
