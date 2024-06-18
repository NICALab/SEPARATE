import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def calc_tsne(latent_features):
    """ Calculate T-SNE feautres of the latent features

    Args:
        latent_features (numpy ndarray, [N, L]): latent feature vector extracted from ClassifyNet
    """
    N = latent_features.shape[0]

    tsne = TSNE(n_components=2) 
    tsne_features = np.array(tsne.fit_transform(latent_features.reshape(N, -1)))
    
    return tsne_features

def calc_PCA(latent_features):
    """ Calculate PCA feautres of the latent features

    Args:
        latent_features (numpy ndarray, [N, L]): latent feature vector extracted from ClassifyNet
    """
    N = latent_features.shape[0]

    pca = PCA(n_components=2)
    PCA_features = np.array(pca.fit_transform(latent_features.reshape(N, -1)))
    
    return PCA_features
    
def twodim_visualization(twodim_features, image_labels, opt, epoch, reduction="tsne"):
    if reduction not in ["tsne", "pca"]:
        raise Exception("reduction must be one of 'tsne' and 'pca'")
    
    # colorcode = cycler(['#9DE500', '#FF4CFF', '#9D9DFF', '#00FFFF', '#FF1919', '#FFFF7F', '#FFCC00', '#FF7F00', '#6666FF', '#00FF00'])
    # colorcode = ['#9DE500', '#FF4CFF', '#9D9DFF', '#00FFFF', '#FF1919', '#FF7F7F', '#FFCC00', '#FF7F00', '#6666FF', '#00FF00']
    colorcode = ["#FF7F00", "#FF4CFF", "#FF1919", "#FFCC00", "#B2B2FF", "#FF9999", "#00FFFF", "#6666FF", "#00FF00", "#B3E600"]
    
    plt.figure(figsize=(8,8))
    image_classes = list(set(list(image_labels)))
    for image_class in image_classes:
        protein_name = opt.protein_list[image_class]
        idx = np.where(image_labels == image_class)[0]
        plt.scatter(twodim_features[idx,0], twodim_features[idx,1], 8, c=colorcode[image_class], \
            marker='.', linewidths=1, label=protein_name)
    plt.legend()
    plt.title("epoch {}".format(epoch), fontdict={'fontweight': 'bold', 'fontsize': 15})
    plt.savefig("{}/tsne/{}/test_epoch_{}_{}.png".format(opt.document_dir, opt.exp_name, epoch, reduction))
    plt.close()
    

if __name__=="__main__":
    pass