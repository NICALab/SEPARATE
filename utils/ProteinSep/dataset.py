import torch
import numpy as np
import skimage.io as skio

from torch.utils.data import Dataset, DataLoader
from utils.image_normalize import return_norm_func

class DatasetProteinSep(Dataset):
    def __init__(self, images, patch_size:list[int], batch_size:int, batch_num:int, rng_seed:int, 
                 min_brightness:float, brightness_range:list[float], norm_type:str, norm_quantile:list[float]):
        """ Dataset for ProteinSepNet

        Args:
            images (list[torch.tensor]): List of the image stack with image stack dimension [C,Z,X,Y]
                                         each channel represents each protein and composite image
            patch_size (list[int]): Size of the image patch, [z,x,y]
            batch_size (int): Size of the batch for each iteration
            batch_num (int): The number of batches for each epoch
            rng_seed (int): The seed of the numpy random number generator
            min_brightness (float): _description_
            brightness_range (list[float]): _description_
            norm_type (str): Type of the normalization
            norm_quantile (list[float]): Upper and lower qunatile for normalization
        """
        # Check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        
        # Initialize
        self.data_weight = list()
        for image in images:
            self.data_weight.append(torch.numel(image[0]))
            
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.patch_rng = np.random.default_rng(rng_seed)
        self.min_brightness = min_brightness
        self.brightness_range = brightness_range
        self.norm_func = return_norm_func(norm_type)
        self.norm_quantile = norm_quantile
        
        # Initialize (normalize)
        self.images_c1 = list()
        self.images_c2 = list()
        self.images_input = list()
        self.images_wobg_c1 = list()
        self.images_wobg_c2 = list()
        self.images_wobg_input = list()
        for idx, image in enumerate(images):
            image_c1, image_c2, image_input = image
            # normalize channel 1
            image_c1, image_wobg_c1 = self.norm_func(image_c1, *norm_quantile)
            self.images_c1.append(image_c1)
            self.images_wobg_c1.append(image_wobg_c1)
            # normalize channel 2
            image_c2, image_wobg_c2 = self.norm_func(image_c2, *norm_quantile)
            self.images_c2.append(image_c2)
            self.images_wobg_c2.append(image_wobg_c2)
            # normalize channel 3 (mixed image)
            image_input, image_wobg_input = self.norm_func(image_input, *norm_quantile)
            self.images_input.append(image_input)
            self.images_wobg_input.append(image_wobg_input)
        
        
    def __len__(self):
        num_batch = 0
        for image in self.images_c1:
            num_batch += int(np.prod(image.shape) / np.prod(self.patch_size))
        return num_batch
    
    def __getitem__(self, index):
        # slicing
        try:
            ds_idx = self.patch_rng.choice(len(self.data_weight), 1)[0]
            z_idx = self.patch_rng.integers(0, self.images_c1[ds_idx].size()[0]-self.patch_size[0]+1)
            x_idx = self.patch_rng.integers(0, self.images_c1[ds_idx].size()[1]-self.patch_size[1]+1)
            y_idx = self.patch_rng.integers(0, self.images_c1[ds_idx].size()[2]-self.patch_size[2]+1)
        except:
            breakpoint()

        # input dataset range
        z_range = slice(z_idx, z_idx + self.patch_size[0])
        x_range = slice(x_idx, x_idx + self.patch_size[1])
        y_range = slice(y_idx, y_idx + self.patch_size[2])

        # patch
        image_c1 = self.images_c1[ds_idx][z_range, x_range, y_range]
        image_c2 = self.images_c2[ds_idx][z_range, x_range, y_range]
        image_input = self.images_input[ds_idx][z_range, x_range, y_range]
        image_wobg_c1 = self.images_wobg_c1[ds_idx][z_range, x_range, y_range]
        image_wobg_c2 = self.images_wobg_c2[ds_idx][z_range, x_range, y_range]
        image_wobg_input = self.images_wobg_input[ds_idx][z_range, x_range, y_range]
        
        # add two images (alpha + beta = 0.8~1.2, Each coeffcient must be larger than min_brightness)
        alpha_beta = self.patch_rng.uniform(*self.brightness_range, size=1) # sum of alpha and beta
        alpha = self.patch_rng.uniform(self.min_brightness, alpha_beta-self.min_brightness, size=1)
        beta = alpha_beta - alpha
        
        network_input = alpha[0] * image_c1 + beta[0] * image_c2
        network_target_1 = alpha[0] * image_wobg_c1
        network_target_2 = beta[0] * image_wobg_c2
        
        synthetic_data = (network_input, network_target_1, network_target_2)
        real_data = (image_input, image_wobg_c1, image_wobg_c2)
        
        return synthetic_data, real_data


def DataLoaderProteinSep(image_file_list:list[str], protein_list:list[str], patch_size:list[int], batch_size:int, batch_num:int, rng_seed:int, 
                         min_brightness:float, brightness_range:list[float], norm_type:str, norm_quantile:list[float], shuffle=True):
    """_summary_

    Args:
        image_file_list (list[str]): List of path to the protein images
        protein_list (list[str]): List of protein for separation
        patch_size (list[int]): Size of the image patch, [z,x,y]
        batch_size (int): Size of the batch for each iteration
        batch_num (int): The number of batches for each epoch
        rng_seed (int): The seed of the numpy random number generator
        min_brightness (float): _description_
        brightness_range (list[float]): _description_
        norm_type (str): Type of the normalization
        norm_quantile (list[float]): Upper and lower qunatile for normalization
        shuffle (bool, optional): Whether shuffle the data or not. Defaults to True.
    """
    
    avg_pool = torch.nn.AvgPool2d(kernel_size=2)
    
    images = list()
    for idx, image_file in enumerate(image_file_list):
        image_c1 = torch.from_numpy(skio.imread("{}_ch1.tif".format(image_file)).astype(np.float32))[1:-1]
        image_c2 = torch.from_numpy(skio.imread("{}_ch2.tif".format(image_file)).astype(np.float32))[1:-1]
        image_input = torch.from_numpy(skio.imread("{}_ch3.tif".format(image_file)).astype(np.float32))[1:-1]

        print(image_input.shape)
            
        image = torch.stack((image_c1, image_c2, image_input), dim=0)
        images.append(image)
        
    dataset = DatasetProteinSep(images, patch_size, batch_size, batch_num, rng_seed, 
                                min_brightness, brightness_range, norm_type, norm_quantile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


if __name__=="__main__":
    pass