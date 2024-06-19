import torch

class FeatureExtractNet(torch.nn.Module):
    def __init__(self, image_patch_size:int, nColor=1, mid_channels=[8,12,16,32,48], out_features=128):
        """ Feature Extraction Network
        
        Args:
            image_patch_size (int): the size of the input image patch to determine the input features of the linear layer.
            nColor (int, optional): the number of channels of the input image patch. Defaults to 1.
            mid_channels (list[int], optional): the number of middle channels of the encoders. Defaults to [8,12,16,32,48].
            out_features (int, optional): the dimension of the fianl output feature vectors. Defaults to 128.
        """
        super(FeatureExtractNet, self).__init__()
        
        self.in_channel = nColor
        self.mid_channels = mid_channels
        self.out_features = out_features
        
        kernel, stride, padding = 3, 1, 1
        
        # encoders
        self.encoders = list()
        for idx, mid_channel in enumerate(mid_channels):
            encoder_in = nColor if idx == 0 else mid_channels[idx-1]
            self.encoders.append(torch.nn.Conv2d(encoder_in, mid_channel, kernel, stride, padding, bias=True))            
        self.encoders = torch.nn.ModuleList(self.encoders)
        
        # linear layer
        self.linear_infeatures = mid_channels[-1] * (image_patch_size // 2 ** len(mid_channels)) ** 2
        self.linear = torch.nn.Linear(self.linear_infeatures, out_features)
        
        # maxpool
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # activation
        self.relu = torch.nn.ReLU()
        self.lrelu = torch.nn.LeakyReLU()
        
    def forward(self, x):
        """ Forward propagation of the FeatureExtractNet

        Args:
            x (torch.tensor): input image stacks with dimension [N,nColor,X,Y] or [nColor,X,Y]
            
        Returns:
            x (torch.tensor): extracted feature vectors with dimension [N,out_features] or [out_features]
        """
        for encoder in self.encoders:
            x = self.maxpool(self.relu(encoder(x)))
        
        x = self.linear(x.reshape(-1, self.linear_infeatures))

        return x


if __name__=="__main__":
    from torchinfo import summary

    model = FeatureExtractNet(256)
    
    network_input = torch.rand([1,1,256,256])
    network_output = model(network_input)
    
    summary(model, [1,1,256,256])
    breakpoint()
    
    model = FeatureExtractNet()
    network_input = torch.zeros([1,1,256,256])
    network_output = model(network_input)
