import os
import glob
import yaml
import argparse

from utils.load_namespace import load_namespace

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    # Experiment profile
    parser.add_argument("--exp_name", type=str, default="myEXP", help="name of the experiment")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for rng")
    parser.add_argument("--results_dir", type=str, default="/media/HDD1/SEPARATE/results/FeatureExtract", help="root directory to save results")
    parser.add_argument("--document_dir", type=str, default="./results/FeatureExtract", help="root directory to save log and namesapce")
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default=["/media/HDD1/SEPARATE/FeatureExtract"], nargs="+", help="root directory to the data")
    parser.add_argument("--protein_list", type=str, default=["GFAP", "LaminB1", "MAP2", "MBP", "NF200", "NeuN"], nargs="+", help="list of the protein for classification")
    parser.add_argument("--testdata_list", type=str, default=[""], nargs="+", help="list of path to the test data")
    parser.add_argument("--norm_type", type=str, default="zslice", help="type of normalization")
    parser.add_argument("--norm_quantile", type=float, default=[0.10, 0.99], nargs="+", help="lower and upper quantile used for normalization")
    
    # Train
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (need epoch-1 model)")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--patch_size", type=int, default=[1,256,256], nargs="+", help="size of the patches")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--batch_num", type=int, default=1000, help="the number of batches for each epoch")
    parser.add_argument("--brightness_range", type=float, default=[0.1,0.9], nargs="+", help="range of brightness for each channel")
    
    parser.add_argument("--lr", type=float, default=3e-3, help="adam: learning rate")
    parser.add_argument("--l1_l2_loss_coef", type=float, default=[0.5,0.5], nargs="+", help="L1/L2 loss coefficients")
    parser.add_argument("--inter_intra_loss_coef", type=float, default=[0.5,0.5], nargs="+", help="inter/intra class loss coefficients")
    parser.add_argument("--inter_margin", type=float, default=0.5, help="inter class loss margin")
    parser.add_argument("--intra_margin", type=float, default=1.5, help="intra class loss margin")
    
    parser.add_argument("--mid_channels_FeatureExtractNet", type=int, default=[8,12,16,32,48], nargs="+", help="the number of channels of ClassifyNet")
    parser.add_argument("--out_features_FeatureExtractNet", type=int, default=128, help="the number of channels of ClassifyNet")
    
    # Util
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--logging_interval_batch", type=int, default=50, help="interval between logging info (in batches)")
    parser.add_argument("--logging_interval", type=int, default=1, help="interval between logging info (in epochs)")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints (in epochs)")
    parser.add_argument("--test_interval", type=int, default=5, help="interval between test checkpoints (in epochs)")
    
    opt = parser.parse_args()
    
    #
    opt.exp_name += "_patch_{}_batch_{}".format(opt.patch_size[-1], opt.batch_size)
    # opt.protein_list = ['CALB2', 'Calnexin', 'GFAP', 'LaminB1', 'NeuN', 'Nucleoin', 'PV', 'S100B']
    opt.protein_list = sorted(opt.protein_list)
    
    # Namespace directory
    os.makedirs("{}/namespace".format(opt.document_dir), exist_ok=True)
    namespace_file = "{}/namespace/{}.yaml".format(opt.document_dir, opt.exp_name)
    
    try:
        # Load Namespace
        breakpoint()
        opt = load_namespace(namespace_file, opt)
    except:
        # Split train and test data
        opt.traindata_list = list()
        opt.testdata_list = list()
        for folder in opt.data_dir:
            opt.traindata_list += glob.glob("{}/train/*.tif".format(folder))
            opt.testdata_list += glob.glob("{}/test/*.tif".format(folder))
        
        opt.traindata_list = [data for data in opt.traindata_list if data not in opt.testdata_list]
        opt.traindata_list = sorted(list(set(opt.traindata_list)))
        opt.testdata_list = sorted(list(set(opt.testdata_list)))
        
        if len(opt.traindata_list) == 0:
            breakpoint()
            raise Exception("")
        
        # Save Namespace
        with open(namespace_file, 'w') as f:
            yaml.dump(vars(opt), f)
            
    return opt
            
    
if __name__=="__main__":
    parse_arguments()
