import os
import glob

import h5py
import tifffile

import torch
import numpy as np

from skimage import io


if __name__=="__main__":
    ##### Image file reorganization (.ims → .tif of each channel) for train data #####
    # Change with your data
    data_folder = "./data/demo_data"
    save_folder = "./data/demo_data_tif"
    #
    # .ims file with 4 channels in data folder named "*{protein α}_{protein β}_{sample idx}.ims"
    # each channel correspond to
    # channel 0     DAPI image for registration marker
    # channel 1     individual image of protein α (CALB2)
    # channel 2     individual image of protein β (Calnexin)
    # channel 3     mixed image of protein α and protein β
    #
    # save channel 1-3 images to .tif file with suffix "ch*"
    # channel 1     *{protein α}_{protein β}_{sample idx}_ch1.tif
    # channel 2     *{protein α}_{protein β}_{sample idx}_ch2.tif
    # channel 3     *{protein α}_{protein β}_{sample idx}_ch3.tif
    ###################################################################
    
    os.makedirs(save_folder, exist_ok=True)

    for filepath in glob.glob("{}/*.ims".format(data_folder)):
        f = h5py.File(filepath, "r")
        for ch in range(3):
            ch += 1
            data = torch.from_numpy(f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]["Channel {}".format(ch)]["Data"][:].astype(np.float32))
            
            filename = filepath.split("/")[-1]
            if ch == 1:
                filename = filename.replace(".ims", "_ch1_GFAP_NeuN.tif")
            elif ch == 2:
                filename = filename.replace(".ims", "_ch2_LaminB1_PV.tif")
            else:
                filename = filename.replace(".ims", "_ch3_Double-cortin_Nucleolin.tif")
                
            tifffile.imwrite("{}/{}".format(save_folder, filename), data.numpy())
