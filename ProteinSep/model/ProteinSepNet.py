import torch

class ProteinSepNet(torch.nn.Module):
    def __init__(self, nTarget=2, mid_channels=[8,12,16,32,48,64], one_by_one_channels=[6,4]):
        """ Protein Separation Network, simple U-Net style

        Args:
            nTarget (int, optional): the number of channels of the output image patch.
                                     the number of target proteins to unmix. Defaults to 2.
            mid_channels (list[int], optional): the number of middle channels. Defaults to [8,12,16,32,48,64].
            one_by_one_channels (list[int], optional): the number of channels of 1x1 conv. Defaults to [6,4].
        """
        super(ProteinSepNet, self).__init__()
        
        self.in_channel = 1
        self.out_channel = nTarget
        self.mid_channels = mid_channels
        self.one_by_one_channels = one_by_one_channels
        
        kernel, stride, padding = 3, 1, 1
        
        # encoders, decoders, upconvs
        self.encoders = list()
        self.decoders = list()
        self.upconvs = list()
        for idx, mid_channel in enumerate(mid_channels):
            # encoder
            encoder_in = self.in_channel if idx == 0 else mid_channels[idx-1]
            self.encoders.append(torch.nn.Conv3d(encoder_in, mid_channel, kernel, stride, padding))
            # decoder
            decoder_out = nTarget if idx == 0 else mid_channel
            self.decoders.append(torch.nn.Conv3d(2*mid_channel, mid_channel, kernel, stride, padding))
            # upconv
            if idx == 1:    # upsample in z
                upconv_kernel = [2,2,2]
                upconv_stride = [2,2,2]
            else:
                upconv_kernel = [1,2,2]
                upconv_stride = [1,2,2]
            upconv_in = mid_channel if idx == len(mid_channels)-1 else mid_channels[idx+1]
            self.upconvs.append(torch.nn.ConvTranspose3d(upconv_in, mid_channel, upconv_kernel, upconv_stride))
        self.encoders = torch.nn.ModuleList(self.encoders)
        self.decoders = torch.nn.ModuleList(self.decoders)
        self.upconvs = torch.nn.ModuleList(self.upconvs)

        # 1x1 convs
        kernel, stride, padding = 1, 1, 0
        self.one_by_one_convs = list()
        for idx, one_by_one_channel in enumerate(one_by_one_channels):
            one_by_one_conv_in = mid_channels[0] if idx == 0 else one_by_one_channels[idx-1]
            self.one_by_one_convs.append(torch.nn.Conv3d(one_by_one_conv_in, one_by_one_channel, kernel, stride, padding))
        self.one_by_one_convs.append(torch.nn.Conv3d(one_by_one_channels[-1], self.out_channel, kernel, stride, padding))
        self.one_by_one_convs = torch.nn.ModuleList(self.one_by_one_convs)
        
        # maxpool
        self.maxpool = torch.nn.MaxPool3d([1,2,2])
        self.maxpoolz = torch.nn.MaxPool3d([2,2,2]) # maxpool in z
        
        # activation
        self.relu = torch.nn.ReLU()
        self.lrelu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        """ Forward propagation of the ProteinSepNet

        Args:
            x (torch.tensor): input image stacks with dimension [N,1,Z,X,Y] or [1,Z,X,Y]

        Returns:
            x (torch.tensor): unmixed image stacks with dimension [N,nTarget,Z,X,Y] or [nTarget,Z,X,Y]
        """
        x_skip_connection = list()
        for idx, encoder in enumerate(self.encoders):
            x = self.relu(encoder(x))
            x_skip_connection.append(x)
            if idx == 1:    # maxpool in z
                x = self.maxpoolz(x)
            else:
                x = self.maxpool(x)
        
        for idx, (decoder, upconv, x_tmp) in enumerate(reversed(list(zip(self.decoders, self.upconvs, x_skip_connection)))):
            try:
                if len(x.shape) == 5:
                    x = torch.concat([upconv(x), x_tmp], dim=1)
                elif len(x.shape) == 4:
                    x = torch.concat([upconv(x), x_tmp], dim=0)
                else:
                    breakpoint()
                x = self.relu(decoder(x))
            except:
                breakpoint()

        for idx, one_by_one_conv in enumerate(self.one_by_one_convs):
            x = one_by_one_conv(x)
            if idx != len(self.one_by_one_convs) - 1:
                x = self.relu(x)
        
        return x


if __name__=="__main__":
    pass
