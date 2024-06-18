import torch
import numpy as np
import skimage.io as skio

from torch.utils.data import Dataset, DataLoader
from utils.image_normalize import return_norm_func

class DatasetFeatureExtract(Dataset):
    def __init__(self, images, image_labels, patch_size:list[int], batch_size:int, batch_num:int, 
                 rng_seed:int, brightness_range:list[float], norm_type:str, norm_quantile:list[float]):
        """ Dataset for FeatureExtractNet

        Args:
            images (list[torch.tensor]): List of the image stack with image stack dimension [C,Z,X,Y]
            image_labels (list[int]): List of the image label corresponds to the protein type of the image stack
            patch_size (list[int]): Size of the image patch, [z,x,y]
            batch_size (int): Size of the batch for each iteration
            batch_num (int): The number of batches for each epoch
            rng_seed (int): The seed of the numpy random number generator
            brightness_range (list[float]): _description_
            norm_type (str): Type of the normalization
            norm_quantile (list[float]): Upper and lower qunatile for normalization
        """
        
        # Check arguments
        if len(patch_size) != 3:
            raise Exception("Length of patch_size must be 3")
        
        # Initialize
        self.data_weight = list()
        for image in images:
            self.data_weight.append(torch.numel(image))
        
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.patch_rng = np.random.default_rng(rng_seed)
        self.brightness_range = brightness_range
        self.norm_func = return_norm_func(norm_type)
        
        # Initialize (normalize)
        self.images = list()
        self.images_wobg = list()
        for idx, image in enumerate(images):
            image, image_wobg = self.norm_func(image, *norm_quantile)
            self.images.append(image)
            self.images_wobg.append(image_wobg)
        
        self.image_labels = image_labels
    
    def __len__(self):
        # num_batch = 0
        # for image in self.images:
        #     num_batch += int(np.prod(image.shape) / np.prod(self.patch_size))
        # return num_batch
        return self.batch_size * self.batch_num
    
    def __getitem__(self, index):
        try:
            # slicing
            ds_idx = self.patch_rng.choice(len(self.data_weight), 1)[0]
            z_idx = self.patch_rng.integers(0, self.images[ds_idx].size()[0]-self.patch_size[0]+1)
            x_idx = self.patch_rng.integers(0, self.images[ds_idx].size()[1]-self.patch_size[1]+1)
            y_idx = self.patch_rng.integers(0, self.images[ds_idx].size()[2]-self.patch_size[2]+1)

            # input dataset range
            z_range = slice(z_idx, z_idx + self.patch_size[0])
            x_range = slice(x_idx, x_idx + self.patch_size[1])
            y_range = slice(y_idx, y_idx + self.patch_size[2])

            # patch
            image = self.images[ds_idx][z_range, x_range, y_range]
            image_wobg = self.images_wobg[ds_idx][z_range, x_range, y_range]
            
            image_label = self.image_labels[ds_idx]
            
        except:
            breakpoint()
            
        return image, image_wobg, image_label


def DataLoaderFeatureExtract(image_file_list:list[str], protein_list:list[str], patch_size:list[int], batch_size:int, batch_num:int, 
                             rng_seed:int, brightness_range:list[float], norm_type:str, norm_quantile:list[float], shuffle=True):
    """ Dataloader for FeatureExtractNet

    Args:
        image_file_list (list[str]): List of path to the protein images
        protein_list (list[str]): List of protein for classification
        patch_size (list[int]): Size of the image patch, [z,x,y]
        batch_size (int): Size of the batch for each iteration
        batch_num (int): The number of batches for each epoch
        rng_seed (int): The seed of the numpy random number generator
        brightness_range (list[float]): _description_
        norm_type (str): Type of the normalization
        norm_quantile (list[float]): Upper and lower qunatile for normalization
        shuffle (bool, optional): Whether shuffle the data or not. Defaults to True.
    """
    
    images = list()
    image_labels = list()
    
    for image_file in image_file_list:
        # image
        image = torch.from_numpy(skio.imread(image_file).astype(np.float32))[1:-1]
        images.append(image)
        
        # image label
        if image_file.split("/")[-1].split("_")[-1] == "ch1.tif":
            protein = image_file.split("/")[-1].split("_")[1]
        elif image_file.split("/")[-1].split("_")[-1] == "ch2.tif":
            protein = image_file.split("/")[-1].split("_")[2]
        else:
            raise Exception("")
        image_labels.append(protein_list.index(protein))
    
    dataset = DatasetFeatureExtract(images, image_labels, patch_size, batch_size, batch_num, 
                                    rng_seed, brightness_range, norm_type, norm_quantile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
    