import os
import glob

import h5py
import tifffile

import torch
import numpy as np

from skimage import io


if __name__=="__main__":
    ##### Image file reorganization for 3-dimesional demonstration (.ims → .tif of each channel) #####
    data_dir = "/media/HDD4/SEPARATE/data/240223_3channel_demo"
    
    # increase z-step size from 0.5um to 1um
    avg_pool = torch.nn.AvgPool3d(kernel_size=[2,1,1])
    
    for filepath in glob.glob("{}/*.ims".format(data_dir)):
        f = h5py.File(filepath, "r")
        for ch in range(3):
            data = torch.from_numpy(f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel {}".format(ch+1)]["Data"][:].astype(np.float32))
            try:
                binned_data = avg_pool(data.unsqeeze(0)).squeeze()
            except:
                binned_data = avg_pool(data[:-1].unsqueeze(0)).squeeze()
            
            filename = filepath.split("/")[-1]
            filename = filename.replace("AI multiplexing 3 channel demo", "3channel_demo")
            filename = filename.replace(".ims", ".tif")
            savefolder = "{}_tif/ch{}".format(data_dir, ch+1)
            os.makedirs(savefolder, exist_ok=True)
            
            tifffile.imwrite("{}/{}".format(savefolder, filename), binned_data.numpy())