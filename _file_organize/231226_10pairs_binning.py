import os
import glob
import tifffile

import torch
import numpy as np

from skimage import io as skio


if __name__=="__main__":
    ##### Image file reorganization for validation data #####
    data_dir = "/media/HDD4/SEPARATE/data/231226_10pairs_tif_denoised"
    
    # 2048 x 2048 images to 1024 x 1024 images
    avg_pool = torch.nn.AvgPool3d(kernel_size=[1,2,2])
    
    for filepath in glob.glob("{}/*.tif".format(data_dir)):
        # remove first and last z-slice of SUPPORT denoised data
        data = torch.from_numpy(skio.imread(filepath)[1:-1].astype(np.float32))
        binned_data = avg_pool(data.unsqueeze(0)).squeeze()
        
        # rename the image file
        # : *{protein α}_{protein β}_*_ch1.tif    mixed image of protein α and protein β
        # : *{protein α}_{protein β}_*_ch2.tif    individual image of protein α
        # : *{protein α}_{protein β}_*_ch3.tif    individual image of protein β
        filename = filepath.split("/")[-1]
        filename = filename.replace("AI multiplexing_", "")
        filename = filename.replace(" pair", "Pair")
        filename = filename.replace("Doublecortin", "DoubleCortin")
        filename = filename.replace(".ims_channel_", "_ch")
        filename = filename.split("_")
        filename = "_".join(filename[:3]+filename[4:])
        savefolder = "{}_binning/{}".format(data_dir, "_".join(filename.split("_")[:3]))
        os.makedirs(savefolder, exist_ok=True)
        
        tifffile.imwrite("{}/{}".format(savefolder, filename), binned_data.numpy())