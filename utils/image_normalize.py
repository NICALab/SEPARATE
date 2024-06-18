import numpy as np
import torch

def return_norm_func(norm_type):
    """ Return the normalization function

    Args:
        norm_type (str): Type of the normalization (one of the 'zstack' or 'zslice').

    Returns:
        norm_func (fucntion): Function that corresponds to the argument norm_type.
    """
    
    if norm_type == 'zstack':
        norm_func = normalize_zstack
    
    elif norm_type == 'zslice':
        norm_func = normalize_zslice
    
    else:
        raise Exception("Type of the normalization must be one of the 'zstack' or 'zslice'")
    
    return norm_func

def normalize_zstack(image, lower_q=0.10, upper_q=0.99):
    """ Normalize the image for entire z-stack

    Args:
        image (torch.tensor): Image stack with dimension [Z,X,Y].
        lower_q (float, optional): Lower quantile used for normalization. Defaults to 0.10.
        upper_q (float, optional): Upper quantile used for normalization. Defaults to 0.99.

    Returns:
        image (torch.tensor): Normalized image stack with dimension [Z,X,Y].
        image_wobg (torch.tensor): Normalized image stack without background with dimension [Z,X,Y].
    """
    # calculate lower quantile and upper quantile, and normalize to [lower quantile: 0, upper quantile: 1]
    # background removal truncated with low quantile
    
    lower_bound = np.quantile(image, lower_q)
    upper_bound = np.quantile(image, upper_q)
    
    image = (image - lower_bound) / (upper_bound - lower_bound)
    try:
        image_wobg = torch.clamp(image, min=0)
    except:
        image_wobg = torch.clamp(torch.from_numpy(image), min=0)
        image_wobg = image_wobg.numpy()
    
    return image, image_wobg

def normalize_zslice(image, lower_q=0.10, upper_q=0.99):
    """ Normalize the image for each z-slice

    Args:
        image (torch.tensor): Image stack with dimension [Z,X,Y].
        lower_q (float, optional): Lower quantile used for normalization. Defaults to 0.10.
        upper_q (float, optional): Upper quantile used for normalization. Defaults to 0.99.

    Returns:
        image (torch.tensor): Normalized image stack with dimension [Z,X,Y].
        image_wobg (torch.tensor): Normalized image stack without background with dimension [Z,X,Y].
    """
    # for each plane, calculate lower quantile and upper quantile, and normalize to [lower quantile: 0, upper quantile: 1]
    # background removal truncated with low quantile
    
    for z in range(image.shape[0]):
        lower_bound = np.quantile(image[z,:,:], lower_q)
        upper_bound = np.quantile(image[z,:,:], upper_q)
        
        image[z,:,:] = (image[z,:,:] - lower_bound) / (upper_bound - lower_bound)
        
    image_wobg = torch.clamp(image, min=0)
    
    return image, image_wobg


if __name__=="__main__":
    pass