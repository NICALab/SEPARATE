# SEPARATE
### Spatial Expression PAttern-guided paiRing And unmixing of proTEins




## Installation
#### Clone the repository
```
git clone https://github.com/NICALab/SEPARATE.git
```
#### Navigate into the SEPARATE folder
```
cd ./SEPARATE
```
#### Creat conda environment
```
conda env create -f environment.yaml
```
#### Activate conda environment
```
conda activate SEPARATE
```
#### Install Pytorch: <https://pytorch.org/get-started/locally/>
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


## Getting started
### 0. Organize the image data
* ##### To train the feature extraction network #####
  
  The folder `./data/sample_data_FeatureExtract/` contains two subfolders named `train` and  `test`, each containing the `.tif` files of individual protein images for training and testing (to identify protein pairing) the _feature extraction network_, respectively.

  * The images are named using the format `{protein name}_{sample idx}.tif`, such as `CALB2_1.tif`, `GFAP_4.tif`, or `PV_3.tif`, and each `.tif` file contains single channel _**[Z, X, Y]**_ image of the protein.

  The name of the images can start with any prefix but must end with the format `{protein name}_{sample idx}.tif` to be compatible with the provided code.
  
* ##### To train the protein separation network #####

  For the pair of two proteins—protein α and protein β—the folder `./data/sample_data_ProteinSep/{protein α}_{protein β}/` also contains two subfolders named `train` and  `test`, each containing the `.tif` files of individual protein images for training and testing the _protein separation network_, respectively.

  * The images containing _**individual sigal of protein α**_ are named using the format `{protein α}_{protein β}_{sample idx}_ch1.tif`.

  * The images containing _**individual sigal of protein β**_ are named using the format `{protein α}_{protein β}_{sample idx}_ch2.tif`.

  * The images containing _**mixed sigal of protein α and protein β**_ are named using the format `{protein α}_{protein β}_{sample idx}_ch3.tif`.
 
  * Each `.tif` file contains single channel _**[Z, X, Y]**_ image

  The name of the folder and images can start with any prefix but must end with the format. For instance, `./data/sample_data_ProteinSep/Group1Pair4_LaminB1_PV/Group1Pair4_LaminB1_PV_1_ch1.tif`

* ##### Demonstration of SEPARATE #####

  For the demonstration of SEPARATE for 2N proteins using N fluorophores, the folder `./data/sample_data_demo/` contains N subfolder named `ch1`, `ch2`, ... , `chN`, each containing the single channel _**[Z, X, Y]**_ images of each pair of two proteins.

  * The images are named `ch{channel number}_{sample idx}.tif `, such as `ch1_1.tif` or `ch3_4.tif`.
 
  There are no restrictions on the names of folders and images, as long as they follow the specified format of N subfolders.

### 1. Train the feature extraction network
```
python -m FeatureExtract.script.train --exp_name mytest_FeatureExtractNet --protein_list CALB2 Calnexin GFAP Double-cortin LaminB1 MAP2 NeuN Nucleolin PV S100B --data_dir ./data/sample_data --results_dir ./results/FeatureExtract/ --n_epochs 100
```
* You can check the extracted feature vector (t-SNE plot) in `./results/FeatureExtract/tsne/mytest_FeatureExtractNet/`
  <img src="https://github.com/NICALab/SEPARATE/assets/88869620/7ef98021-c980-415f-995e-184fe8c5292a.png" height="350"/>

#### 2. Spatial expression pattern guided protein pairing
```
python -m FeatureExtract.script.test ./results/FeatureExtract/namespace/mytest_FeatureExtractNet.yaml  --pairing_protein_list Double-cortin GFAP LaminB1 NeuN Nucleolin PV --test_epoch 10
```
* You can check the optimal protein pairing in the terminal like below

  ![SpatialExpressionPatternGuidedProteinPairing](https://github.com/NICALab/SEPARATE/assets/88869620/9c071038-0017-4d62-8139-b0f29e779db1)

#### 3. Train the protein separation network
```
python -m ProteinSep.script.train --exp_name mytest_ProteinSepNet --protein_list LaminB1 PV --data_dir ./data/sample_data/Group1Pair4_LaminB1_PV --results_dir ./results/ProteinSep/ --n_epochs 10000
```

#### 4. Inference of test data for protein separation network
```
python -m ProteinSep.script.test ./results/ProteinSep/namespace/mytest_ProteinSepNet_LaminB1_PV.yaml --test_epoch 100
```

#### 5. Demonstration of SEPARATE
```
python -m ProteinSep.script.demo ./results/ProteinSep/namespace/mytest_ProteinSepNet_LaminB1_PV.yaml --testdata_dir ./data/demo_data/ch2_LaminB1_PV --test_epoch 100
```
