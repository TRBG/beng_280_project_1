import torch
from torch import nn

__all__ = ['CT_Recon_Net']


#######################################
# The Building Blocks of the Network
class MainConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
      
      
class FirstConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out
     
    
#######################################
# The Main Network
class CT_Recon_Net(nn.Module):
    def __init__(self, input_channels = 12, **kwargs):
        super().__init__()

        num_filter = 64
        
        self.conv_start = FirstConvBlock(input_channels, num_filter)
        
        self.conv_1 = MainConvBlock(num_filter, num_filter)
        self.conv_2 = MainConvBlock(num_filter, num_filter)
        self.conv_3 = MainConvBlock(num_filter, num_filter)
        self.conv_4 = MainConvBlock(num_filter, num_filter)
        self.conv_5 = MainConvBlock(num_filter, num_filter)
        self.conv_6 = MainConvBlock(num_filter, num_filter)
        self.conv_7 = MainConvBlock(num_filter, num_filter)
        self.conv_8 = MainConvBlock(num_filter, num_filter)
        self.conv_9 = MainConvBlock(num_filter, num_filter)
        self.conv_10 = MainConvBlock(num_filter, num_filter)
        self.conv_11 = MainConvBlock(num_filter, num_filter)
        self.conv_12 = MainConvBlock(num_filter, num_filter)
        self.conv_13 = MainConvBlock(num_filter, num_filter)
        self.conv_14 = MainConvBlock(num_filter, num_filter)
        self.conv_15 = MainConvBlock(num_filter, num_filter)
        
        self.conv_fin = nn.Conv2d(num_filter, 1, 3, padding=1)


    def forward(self, input):
        x = self.conv_start(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.conv_14(x)
        x = self.conv_15(x)
        
        output = self.conv_fin(x)
        
        return output

