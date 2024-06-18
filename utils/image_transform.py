import torch
import numpy as np

def random_transform(input, target1, target2, rng:np.random.default_rng):
    """ Randomly rotate/flip the image (only rotate and flip, XY-direction)

    Args:
        input (torch.tensor): Input image stack with dimension [N,Z,X,Y].
        target1 (torch.tensor): Target image stack with dimension [N,Z,X,Y]. Can be None.
        target2 (torch.tensor): Target image stack with dimension [N,Z,X,Y]. Can be None.
        rng (np.random.default_rng): random number generator

    Returns:
        input (torch.tensor): Randomly rotated/flipped input image stack with dimension [N,Z,X,Y].
        target1 (torch.tensor): Randomly rotated/flipped target image stack with dimension [N,Z,X,Y].
        target2 (torch.tensor): Randomly rotated/flipped target image stack with dimension [N,Z,X,Y].
    """

    rand_num_rotate = rng.integers(0, 4) # random number for rotation
    rand_num_flip = rng.integers(0, 2) # random number for flip
    
    # Rotate
    if rand_num_rotate == 1:
        input = torch.rot90(input, k=1, dims=(2, 3))
        if target1 is not None:
            target1 = torch.rot90(target1, k=1, dims=(2, 3))
            target2 = torch.rot90(target2, k=1, dims=(2, 3))
    elif rand_num_rotate == 2:
        input = torch.rot90(input, k=2, dims=(2, 3))
        if target1 is not None:
            target1 = torch.rot90(target1, k=2, dims=(2, 3))
            target2 = torch.rot90(target2, k=2, dims=(2, 3))
    elif rand_num_rotate == 3:
        input = torch.rot90(input, k=3, dims=(2, 3))
        if target1 is not None:
            target1 = torch.rot90(target1, k=3, dims=(2, 3))
            target2 = torch.rot90(target2, k=3, dims=(2, 3))
    
    # Flip
    if rand_num_flip == 1:
        input = torch.flip(input, dims=[2])
        if target1 is not None:
            target1 = torch.flip(target1, dims=[2])
            target2 = torch.flip(target2, dims=[2])
    
    return input, target1, target2


if __name__=="__main__":
    pass