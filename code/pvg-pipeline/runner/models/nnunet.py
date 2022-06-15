from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List, Union

def init_weights(m):
    if (type(m) == nn.Conv3d or
        type(m) == nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ConvInstanceLeaky(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int,Tuple[int,int,int]]=3,
                 padding: Union[int,Tuple[int,int,int]]=1):
        super(ConvInstanceLeaky, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True, negative_slope=1e-2))
        
    def forward(self, x: Tensor):
        return self.block(x)

class Double(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: Tuple[int,int],
                 kernel_size: Union[int,Tuple[int,int,int]]=3,
                 padding: Union[int,Tuple[int,int,int]]=1):
        super(Double, self).__init__()
        
        self.double_block = nn.Sequential(
            ConvInstanceLeaky(in_channels, out_channels[0], kernel_size, padding),
            ConvInstanceLeaky(out_channels[0], out_channels[1], kernel_size, padding))
                                    
    def forward(self, x: Tensor):
        return self.double_block(x)
        
class Down(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Tuple[int,int],
                 pool_kernel: Union[int,Tuple[int,int,int]],
                 p: float=0.1):
        super(Down, self).__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool3d(pool_kernel),
            nn.Dropout3d(p=p),
            Double(in_channels, out_channels)) 

    def forward(self, x: Tensor):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Tuple[int,int],
                 p: float=0.1):
        super(Up, self).__init__()
        
        self.drop = nn.Dropout(p=p)
        self.conv_up = nn.Conv3d(in_channels, out_channels[0], 3, padding=1, bias=True)
        self.conv_in = nn.Conv3d(in_channels, out_channels[0], 3, padding=1, bias=False)
        self.single_block = nn.Sequential(nn.InstanceNorm3d(out_channels[0], affine=True),
                                   nn.LeakyReLU(inplace=True, negative_slope=1e-2),
                                   ConvInstanceLeaky(out_channels[0], out_channels[1]))
        
    def forward(self, x: Tensor, y: Tensor):
        x = F.interpolate(x, size=y.size()[-3:])
        x, y = self.drop(x), self.drop(y)
        z = self.conv_up(x) + self.conv_in(y)
        return self.single_block(z)
    
class NNUNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 filters: int, 
                 depth: int,
                 p: float=0.1):
        
        super(NNUNet, self).__init__()
        
        # components
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.classifier = None
        
        # build encoder
        self.encoder.append(Double(in_channels, 2*[filters])) # 32
        for layer in range(0, depth-2): # (0, 64), (1, 128), (2, 256)
            self.encoder.append(Down(filters*2**layer, 2*[filters*2**(layer+1)], 2, p))
        self.encoder.append(Down(filters*2**(depth-2), [filters*2**(depth-1), filters*2**(depth-2)], 2, p))
        
        # build decoder
        for layer in range(depth-2, 0, -1): 
            self.decoder.append(Up(filters*2**layer, [filters*2**layer, filters*2**(layer-1)], p))
        self.decoder.append(Up(filters, 2*[filters], p))
        
        # build classifier
        self.classifier = nn.Conv3d(filters, out_channels, 1, padding=0)
                            
    def encode(self, x: Tensor, cache: bool=True):
        if cache: encodings = []
        for layer in range(len(self.encoder)):
            x = self.encoder[layer](x)
            if cache: encodings.append(x)
        if cache: return encodings
        else: return x
                     
    def decode(self, x: List):
        decoding = self.decoder[0](x[-1], x[-2])
        for layer in range(1, len(self.decoder)):
            decoding = self.decoder[layer](decoding, x[self.depth-layer-2])
        return decoding
    
    def forward(self, x: Tensor):
        encoding = self.encode(x)
        decoding = self.decode(encoding)
        output = self.classifier(decoding)
        return output