import torch

class FeatureExtractNet(torch.nn.Module):
    def __init__(self, image_patch, nColor=1, mid_channels=[8,12,16,32,48], out_features=128):
        super(FeatureExtractNet, self).__init__()
        
        self.in_channel = nColor
        self.mid_channels = mid_channels
        self.out_features = out_features
        
        kernel, stride, padding = 3, 1, 1
        
        self.encoders = list()
        for idx, mid_channel in enumerate(mid_channels):
            encoder_in = nColor if idx == 0 else mid_channels[idx-1]
            self.encoders.append(torch.nn.Conv2d(encoder_in, mid_channel, kernel, stride, padding, bias=True))            
        self.encoders = torch.nn.ModuleList(self.encoders)
        
        self.linear_infeatures = mid_channels[-1] * (image_patch // 2 ** len(mid_channels)) ** 2
        self.linear = torch.nn.Linear(self.linear_infeatures, out_features)
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        
        
    def forward(self, x):
        """ Forward propagation of the FeatureExtractNet

        Args:
            x (torch.tensor): Image stack with dimension [N,nColor,X,Y] or [nColor,X,Y]
            
        Returns:
            x (torch.tensor): Extracted feature with dimension [N,out_features] or [out_features]
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
