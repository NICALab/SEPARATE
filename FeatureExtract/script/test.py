import yaml
import argparse
import tifffile

import torch
import numpy as np

from tqdm import tqdm
from scipy.io import savemat

from FeatureExtract.model.FeatureExtractNet import FeatureExtractNet
from utils.FeatureExtract.dimension_reduction import calc_tsne, twodim_visualization
from utils.FeatureExtract.protein_pairing import calc_distance, find_optimal_group_of_pairs
from utils.FeatureExtract.dataset import DataLoaderFeatureExtract
from utils.image_transform import random_transform

def test(dataloader, model, opt):
    """ Test FeatureExtractNet
    
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
        
        for idx, data in enumerate(tqdm(dataloader, desc="Test FeatureExtractNet")):
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
    ##### Initialize #####
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace_file", type=str, help="path to the namespace file")
    parser.add_argument("--pairing_protein_list", type=str, default=[""], nargs="+", help="list of proteins for pairing")
    parser.add_argument("--test_epoch", type=int, default=0, help="epoch to test the model")
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--batch_num", type=int, default=20, help="")
    
    parser.add_argument("--save_test_images", action="store_true", help="")
    test_opt = parser.parse_args()
    
    with open(test_opt.namespace_file) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt = argparse.Namespace(**opt)
    print(opt.exp_name)
    
    if test_opt.pairing_protein_list == [""]:
        test_opt.pairing_protein_list = opt.protein_list
    elif set(test_opt.pairing_protein_list).issubset(set(opt.protein_list)):
        test_opt.pairing_protein_list = sorted(test_opt.pairing_protein_list)
    else:
        raise Exception("pairing_protein_list must be the subset of {}".format(set(opt.protein_list)))
    print("The list of proteins for pairing: ", test_opt.pairing_protein_list)
    
    ##### Load model #####
    model = FeatureExtractNet(opt.patch_size[-1], nColor=1, mid_channels=opt.mid_channels_FeatureExtractNet, 
                              out_features=opt.out_features_FeatureExtractNet)
    model.load_state_dict(torch.load("{}/saved_models/{}/model_{}.pth".\
        format(opt.results_dir, opt.exp_name, test_opt.test_epoch)))
    
    if not opt.use_CPU:
        model = model.cuda()
    print('Loaded pre-trained model weights of epoch {}'.format(test_opt.test_epoch))
    
    ##### Load dataset #####
    print("The number of test dataset: {}".format(len(opt.testdata_list)))
    dataloader_test = DataLoaderFeatureExtract(opt.testdata_list, opt.protein_list, opt.patch_size, test_opt.batch_size, test_opt.batch_num, 
                                               opt.random_seed, opt.brightness_range, opt.norm_type, opt.norm_quantile, shuffle=True)
    
    images, image_labels, latent_features, loss_mean_total, loss_mean_inter, loss_mean_intra \
        = test(dataloader_test, model, opt)
    
    ##### Dimension reduction #####
    tsne_features = calc_tsne(latent_features)
    twodim_visualization(tsne_features, image_labels, opt, opt.epoch, "tsne")
    
    savemat("{}/tsne/{}/test_epoch_{}.mat".format(opt.document_dir, opt.exp_name,test_opt.test_epoch), 
            {"latent_features": latent_features, "tsne_features": tsne_features, 
             "image_labels": image_labels, "protein_list": opt.protein_list})
    
    ##### Calculate feature-based distances #####
    distance_dictionary, distance_matrix = calc_distance(latent_features, image_labels, opt.protein_list, test_opt.pairing_protein_list)
    savemat("{}/feature-based_distance/{}/{}.mat".format(opt.document_dir, opt.exp_name, test_opt.test_epoch), \
        {"distance_dictionary": distance_dictionary, "distance_matrix": distance_matrix, 
         "pairing_protein_list": test_opt.pairing_protein_list})
    
    ##### Spatial expression pattern guided protein pairing #####
    find_optimal_group_of_pairs(distance_dictionary.items(), test_opt.pairing_protein_list)
    
    ##### Save test images #####
    if test_opt.save_test_images:
        for idx, image in enumerate(images):
            tifffile.imwrite("{}/images/{}/epoch_{}_{}.tif".format(opt.results_dir, opt.exp_name, opt.epoch, idx+1), image.astype(np.float16))
