"""Model construction
1. We offer two versions of BsiNet, one concise and the other clear
2. The clear version is designed for user understanding and modification
3. You can use these attention mechanism we provide to bulid a new multi-task model, and you can also
4. You can also add your own module or change the location of the attention mechanism to build a better model
5........................................................
..... Baseline + GLCA + BGM + MSF
"""

from torch.nn.parameter import Parameter
from timm.models.registry import register_model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.pvtv2 import pvt_v2_b2
from tensorboardX import SummaryWriter

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


#SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x



#scce注意力模块
class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


##This is a concise version of the BsiNet whose modules are better packaged

class BsiNet(nn.Module):

    output_downscaled = 1
    module = NetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )

        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        self.sge = SpatialGroupEnhance(32)

        # self.ca1 = ChannelAttention()
        # self.sa1 = SpatialAttention()

        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):

            x_out2 = upsample(x_out)
            x_out= (torch.cat([x_out2, x_skip], 1))
            x_out = up(x_out)

        if self.add_output:

            x_out = self.sge(x_out)

            x_out1 = self.conv_final1(x_out)
            x_out2 = self.conv_final2(x_out)
            x_out3 = self.conv_final3(x_out)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1,dim=1)
                x_out2 = F.log_softmax(x_out2,dim=1)
            x_out3 = torch.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]




class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(LaplaceConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        # Generate Laplace kernel
        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)  ##8领域
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(self.bn(x1))

        return x1


######CBAM注意力
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x



###边界增强模块/边界引导模块（boundary guided module,BGM）
class Boundary_guided_module(nn.Module):
    def __init__(self, in_channel1,in_channel2,out_channel):
        super(Boundary_guided_module, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channel1, out_channel, 1)  ##1x1卷积用来降低通道数
        self.conv2 = nn.Conv2d(in_channel2, out_channel, 1)  ##1x1卷积用来降低通道数
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,edge,semantic):
        x = self.conv1(edge)
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = self.sigmoid(x)
        x = x*self.conv2(semantic)
        x = x + self.conv2(semantic)
        return x


class Long_distance(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(Long_distance, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out

##局部和全局上下文（或近和远距离上下文提取）
class near_and_long(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(near_and_long, self).__init__()
        self.long = Long_distance(in_channel)
        self.near = Residual(in_channel,out_channel)
        self.conv1 = nn.Conv2d(in_channel+out_channel,out_channel,1)

    def forward(self,x):
        x1 = self.long(x)
        x2 = self.near(x)
        fuse = torch.cat([x1,x2], 1)
        fuse = self.conv1(fuse)

        return fuse


class multi_scale_fuseion(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(multi_scale_fuseion, self).__init__()
        self.c1 = Conv(in_channel,out_channel, kernel_size=1, padding=0)
        self.c2 = Conv(in_channel,out_channel, kernel_size=3, padding=1)
        self.c3 = Conv(in_channel,out_channel, kernel_size=7, padding=3)
        self.c4 = Conv(in_channel,out_channel, kernel_size=11, padding=5)
        self.s1 = Conv(out_channel*4,out_channel, kernel_size=1, padding=0)
        self.attention = CBAM(out_channel)

    def forward(self,x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x5 = torch.cat([x1,x2,x3,x4], 1)
        x5 = self.s1(x5)
        x6 = self.attention(x5)

        return x6


class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(Residual, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, padding=1,relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding, bias=bias)
        self.relu = relu
        self.bn = bn
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



###1. Laplace卷积的有效性
class Field1(nn.Module):
    def __init__(self, channel=32,num_classes=2,drop_rate=0.4):
        super(Field1, self).__init__()

        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r"E:\zhaohang\DeepL\code\model_New\preweight\pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.channel = channel
        self.num_classes = num_classes

        self.edge_lap = LaplaceConv2d(in_channels=3,out_channels=1)
        self.conv_lap = nn.Conv2d(3,32,3)
        # self.conv1 = Residual(1,32)
        self.conv2 = nn.Conv2d(64, 16, 1)
        self.attention1 = CBAM(32)
        self.attention2 = CBAM(64)
        self.fuse1 = near_and_long(512,256)
        self.fuse2 = near_and_long(320, 128)
        self.fuse3 = near_and_long(128, 64)
        self.fuse4 = near_and_long(64, 32)
        ###
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        ##
        self.boundary1 = Boundary_guided_module(64,64,32)
        self.boundary2 = Boundary_guided_module(64,64, 16)
        self.boundary3 = Boundary_guided_module(64,128, 16)
        self.boundary4 = Boundary_guided_module(64, 256, 16)
        ##
        self.multi_fusion = multi_scale_fuseion(64,64)
        self.sigmoid = nn.Sigmoid()
        self.out_feature = nn.Conv2d(64,1,1)
        self.edge_feature = nn.Conv2d(64, num_classes, 1)
        #输出距离图
        self.dis_feature = nn.Conv2d(64,1,1)

    def forward(self, x):
        #---------------------------------------------------------------------#
        # models_Lap.py 消融实验------ Laplace卷积的有效性

        # edge = self.edge_lap(x)
        edge = self.conv_lap(x)  ##32
        # edge = self.conv1(edge)

        pvt = self.backbone(x)
        x1 = pvt[0]       ##（b,64,64,64）
        x1 = self.drop(x1)
        x2 = pvt[1]          ##（b,128,32,32）
        x2 = self.drop(x2)
        x3 = pvt[2]        ##（b,320,16,16）
        x3 = self.drop(x3)
        x4 = pvt[3]        ##（b,512,32,32）
        x4 = self.drop(x4)
        ##global and local context aggregation
        x1 = self.fuse4(x1)  #32
        x2 = self.fuse3(x2)  #64
        x3 = self.fuse2(x3)  #128
        x4 = self.fuse1(x4)  #256
        ### boundary guided module
        x1 = self.up1(x1)
        edge = torch.cat([edge,x1],1) #64
        edge = self.attention2(edge)
        edge1 = self.conv2(edge)
        # bs1 = self.boundary1(edge,x1)
        x2 = self.up2(x2)
        bs2 = self.boundary2(edge, x2)
        x3 = self.up3(x3)
        bs3 = self.boundary3(edge, x3)
        x4 = self.up4(x4)
        bs4 = self.boundary4(edge, x4)
        ###multi-scale feature fusion module
        ms = torch.cat([edge1,bs2,bs3,bs4],1)
        out = self.multi_fusion(ms)

        # out = self.boundary1(edge,out)  #加上好像没啥变化，自己可以保留试试


        edge_out = self.edge_feature(edge)
        edge_out = F.log_softmax(edge_out, dim=1)
        mask_out = self.out_feature(out)
        # mask_out = F.log_softmax(mask_out,dim=1)
        #输出dist_out
        dis_out = self.dis_feature(out)


        return [mask_out,edge_out,dis_out]




###2. 去掉整个边界分支—边界分支的有效性
class Field2(nn.Module):
    def __init__(self, channel=32,num_classes=2,drop_rate=0.4):
        super(Field2, self).__init__()

        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r"E:\zhaohang\DeepL\code\model_New\preweight\pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.channel = channel
        self.num_classes = num_classes

        self.edge_lap = LaplaceConv2d(in_channels=3,out_channels=1)
        self.conv_lap = nn.Conv2d(3,32,3)
        # self.conv1 = Residual(1,32)
        self.conv2 = nn.Conv2d(480, 64, 1)
        self.attention1 = CBAM(32)
        self.attention2 = CBAM(64)
        self.fuse1 = near_and_long(512,256)
        self.fuse2 = near_and_long(320, 128)
        self.fuse3 = near_and_long(128, 64)
        self.fuse4 = near_and_long(64, 32)
        ###
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        ##
        self.boundary1 = Boundary_guided_module(64,64,32)
        self.boundary2 = Boundary_guided_module(64,64, 16)
        self.boundary3 = Boundary_guided_module(64,128, 16)
        self.boundary4 = Boundary_guided_module(64, 256, 16)
        ##
        self.multi_fusion = multi_scale_fuseion(64,64)
        self.sigmoid = nn.Sigmoid()
        self.out_feature = nn.Conv2d(64,1,1)
        self.edge_feature = nn.Conv2d(64, num_classes, 1)
        #输出距离图
        self.dis_feature = nn.Conv2d(64,1,1)

    def forward(self, x):
        #---------------------------------------------------------------------#
        #

        # edge = self.edge_lap(x)
        # edge = self.conv_lap(x)  ##32
        # edge = self.conv1(edge)

        pvt = self.backbone(x)
        x1 = pvt[0]       ##（b,64,64,64）
        x1 = self.drop(x1)
        x2 = pvt[1]          ##（b,128,32,32）
        x2 = self.drop(x2)
        x3 = pvt[2]        ##（b,320,16,16）
        x3 = self.drop(x3)
        x4 = pvt[3]        ##（b,512,32,32）
        x4 = self.drop(x4)
        ##global and local context aggregation
        x1 = self.fuse4(x1)  #32
        x2 = self.fuse3(x2)  #64
        x3 = self.fuse2(x3)  #128
        x4 = self.fuse1(x4)  #256
        ### boundary guided module
        x1 = self.up1(x1)
        # edge = torch.cat([edge,x1],1) #64
        # edge = self.attention2(edge)
        # edge1 = self.conv2(edge)
        # bs1 = self.boundary1(edge,x1)
        x2 = self.up2(x2)
        # bs2 = self.boundary2(edge, x2)
        x3 = self.up3(x3)
        # bs3 = self.boundary3(edge, x3)
        x4 = self.up4(x4)
        # bs4 = self.boundary4(edge, x4)
        ###multi-scale feature fusion module
        ms = torch.cat([x1,x2,x3,x4],1)
        ms = self.conv2(ms)
        out = self.multi_fusion(ms)

        # out = self.boundary1(edge,out)  #加上好像没啥变化，自己可以保留试试


        # edge_out = self.edge_feature(edge)
        # edge_out = F.log_softmax(edge_out, dim=1)
        mask_out = self.out_feature(out)
        # mask_out = F.log_softmax(mask_out,dim=1)
        #输出dist_out
        dis_out = self.dis_feature(out)


        return [mask_out,dis_out]


###3. 去掉边界辅助任务—边界任务优化的有效性
class Field3(nn.Module):
    def __init__(self, channel=32,num_classes=2,drop_rate=0.4):
        super(Field3, self).__init__()

        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r"E:\zhaohang\DeepL\code\model_New\preweight\pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.channel = channel
        self.num_classes = num_classes

        self.edge_lap = LaplaceConv2d(in_channels=3,out_channels=1)
        self.conv_lap = nn.Conv2d(3,32,3)
        self.conv1 = Residual(1,32)
        self.conv2 = nn.Conv2d(64, 16, 1)
        self.attention1 = CBAM(32)
        self.attention2 = CBAM(64)
        self.fuse1 = near_and_long(512,256)
        self.fuse2 = near_and_long(320, 128)
        self.fuse3 = near_and_long(128, 64)
        self.fuse4 = near_and_long(64, 32)
        ###
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        ##
        self.boundary1 = Boundary_guided_module(64,64,32)
        self.boundary2 = Boundary_guided_module(64,64, 16)
        self.boundary3 = Boundary_guided_module(64,128, 16)
        self.boundary4 = Boundary_guided_module(64, 256, 16)
        ##
        self.multi_fusion = multi_scale_fuseion(64,64)
        self.sigmoid = nn.Sigmoid()
        self.out_feature = nn.Conv2d(64,1,1)
        self.edge_feature = nn.Conv2d(64, num_classes, 1)
        #输出距离图
        self.dis_feature = nn.Conv2d(64,1,1)

    def forward(self, x):
        #---------------------------------------------------------------------#

        edge = self.edge_lap(x)
        edge = self.conv1(edge)

        pvt = self.backbone(x)
        x1 = pvt[0]       ##（b,64,64,64）
        x1 = self.drop(x1)
        x2 = pvt[1]          ##（b,128,32,32）
        x2 = self.drop(x2)
        x3 = pvt[2]        ##（b,320,16,16）
        x3 = self.drop(x3)
        x4 = pvt[3]        ##（b,512,32,32）
        x4 = self.drop(x4)
        ##global and local context aggregation
        x1 = self.fuse4(x1)  #32
        x2 = self.fuse3(x2)  #64
        x3 = self.fuse2(x3)  #128
        x4 = self.fuse1(x4)  #256
        ### boundary guided module
        x1 = self.up1(x1)
        edge = torch.cat([edge,x1],1) #64
        edge = self.attention2(edge)
        edge1 = self.conv2(edge)
        # bs1 = self.boundary1(edge,x1)
        x2 = self.up2(x2)
        bs2 = self.boundary2(edge, x2)
        x3 = self.up3(x3)
        bs3 = self.boundary3(edge, x3)
        x4 = self.up4(x4)
        bs4 = self.boundary4(edge, x4)
        ###multi-scale feature fusion module
        ms = torch.cat([edge1,bs2,bs3,bs4],1)
        out = self.multi_fusion(ms)

        mask_out = self.out_feature(out)
        # mask_out = F.log_softmax(mask_out,dim=1)
        #输出dist_out
        dis_out = self.dis_feature(out)

        return [mask_out, dis_out]


if __name__ == "__main__":
    tensor = torch.randn((8, 3, 256, 256))
    net = Field1()
    # 打印模型每一层的名字
    for name, module in net.named_modules():
        print(f'Layer Name: {name}')
    outputs = net(tensor)
    for output in outputs:
        print(output.size())

