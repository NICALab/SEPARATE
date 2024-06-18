import yaml
import argparse

def load_namespace(namespace_file:str, opt:argparse.Namespace):
    """ Load namespace

    Args:
        namespace_file (str): Path of the namespac file.
        opt (argparse.Namespace): _description_

    Returns:
        my_opt (argparse.Namespace): _description_
    """
    with open(namespace_file) as f:
        loaded_opt = yaml.load(f, Loader=yaml.FullLoader)
        
    loaded_opt = argparse.Namespace(**loaded_opt)
    
    loaded_opt.epoch = opt.epoch
    loaded_opt.n_epochs = opt.n_epochs
    loaded_opt.use_CPU = opt.use_CPU
    loaded_opt.n_cpu = opt.n_cpu
        
    return loaded_opt


if __name__=="__main__":
    pass