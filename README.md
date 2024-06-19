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
#### 0. Organize the image data to train the network



#### 1. Train the feature extraction network
```
python -m FeatureExtract.script.train --exp_name mytest_FeatureExtractNet --protein_list CALB2 Calnexin GFAP Double-cortin LaminB1 MAP2 NeuN Nucleolin PV S100B --data_dir ./data/sample_data --results_dir ./results/FeatureExtract/ --n_epochs 100
```
```
python -m FeatureExtract.script.train --exp_name mytest_FeatureExtractNet_6proteins ---protein_list GFAP Double-cortin LaminB1 NeuN Nucleolin PV --data_dir ./data/sample_data --results_dir ./results/FeatureExtract/ --n_epochs 100
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

#### 5. Demonstration of the protein separation network
```
python -m ProteinSep.script.demo ./results/ProteinSep/namespace/mytest_ProteinSepNet_LaminB1_PV.yaml --testdata_dir ./data/demo_data/ch2_LaminB1_PV --test_epoch 100
```
