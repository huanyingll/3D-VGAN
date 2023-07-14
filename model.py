import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import params
from torch.autograd import Variable
from thop import profile
'''

model.py

Define our GAN model

The cube_len is 32x32x32, and the maximum number of feature map is 256, 
so the results may be inconsistent with the paper

'''

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """ 空间注意力机制 将通道维度通过最大池化和平均池化进行压缩，然后合并，再经过卷积和激活函数，结果和输入特征图点乘

        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print('x shape', x.shape)
        # (2,512,8,8) -> (2,1,8,8)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (2,512,8,8) -> (2,1,8,8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (2,1,8,8) + (2,1,8,8) -> (2,2,8,8)
        cat = torch.cat([avg_out, max_out], dim=1)
        # (2,2,8,8) -> (2,1,8,8)
        out = self.conv1(cat)
        return x * self.sigmoid(out)

#Residual Block
class ResidualBlock(nn.Module):
    def __init__(self,in_channels=1, out_channels=1):
        """
        残差块
        :param input_channels: 输入通道数
        :param num_channels: 输出通道数
        """
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        X = self.conv3(X)
        Y += X
        return F.relu(Y)
# Attention block
class AttentionBlock(nn.Module):
    def __init__(self, in_channels=1, ratio=4,model_channels=128):
        self.model_channels=model_channels
        """
        注意力机制
        :param in_channels:输入通道数
        :param ratio: 扩大比例
        """
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((model_channels,model_channels,model_channels))
        self.max_pool = nn.AdaptiveMaxPool3d((model_channels,model_channels,model_channels))

        self.conv1 = nn.Conv3d(in_channels, in_channels * ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels * ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        """
        a1=self.avg_pool(x)
        print(a1.shape)
        a2=self.conv1(a1)
        print(a2.shape)
        a3=self.relu1(a2)
        print(a3.shape)
        avg_out=self.conv2(a3)
        print(avg_out.shape)
        """
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class net_G(torch.nn.Module):
    def __init__(self):
        super(net_G, self).__init__()
        self.cube_len = params.cube_len
        self.bias = params.bias
        self.z_dim = params.z_dim
        self.leak_value = params.leak_value
        self.mode=params.mode
        self.f_dim = (int(self.mode/4)+1)*4
        padd = (0, 0, 0)
        if self.cube_len == 128:
            padd = (1, 1, 1)
        self.layer1 = self.conv_layer(self.mode, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias,model_channels=128)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim * 2, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=64)
        self.layer3 = self.conv_layer(self.f_dim * 2, self.f_dim * 4, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=32)
        self.layer4 = self.conv_layer(self.f_dim * 4, self.f_dim * 8, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=16)
        self.layer5 = self.conv_layer(self.f_dim * 8, self.f_dim * 16, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=8)
        self.layer6 = self.conv_layer(self.f_dim * 16, self.f_dim * 32, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=4)
        self.layer7 = self.conv_layer(self.f_dim * 32, self.f_dim * 64, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias,model_channels=2)

        self.layer8 = self.conv_transpose_layer(self.z_dim, self.f_dim * 32, kernel_size=4, stride=4, padding=(1, 1, 1),
                                                bias=self.bias,model_channels=1)
        self.layer9 = self.conv_transpose_layer(self.f_dim * 32, self.f_dim * 16, kernel_size=4, stride=2, padding=padd,
                                                bias=self.bias,model_channels=2)
        self.layer10 = self.conv_transpose_layer(self.f_dim * 16, self.f_dim * 8, kernel_size=4, stride=2, padding=padd,
                                                bias=self.bias,model_channels=4)
        self.layer11 = self.conv_transpose_layer(self.f_dim * 8, self.f_dim * 4, kernel_size=4, stride=2, padding=padd,
                                                bias=self.bias,model_channels=8)
        self.layer12 = self.conv_transpose_layer(self.f_dim * 4, self.f_dim * 2, kernel_size=4, stride=2, padding=padd,
                                                 bias=self.bias,model_channels=16)
        self.layer13 = self.conv_transpose_layer(self.f_dim * 2, self.f_dim, kernel_size=4, stride=2, padding=padd,
                                                 bias=self.bias,model_channels=32)
        self.layer14 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.f_dim, self.mode, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )
        self.fc1 = torch.nn.Linear(self.f_dim * 64, self.z_dim)
        self.fc2 = torch.nn.Linear(self.f_dim * 64, self.z_dim)

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=False,model_channels=128):
        layer = torch.nn.Sequential(
            AttentionBlock(in_channels=input_dim, ratio=4, model_channels=model_channels),
            ResidualBlock(in_channels=input_dim, out_channels=input_dim),
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def conv_transpose_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=False,model_channels=128):
        layer = torch.nn.Sequential(
            AttentionBlock(in_channels=input_dim, ratio=4, model_channels=model_channels),
            ResidualBlock(in_channels=input_dim, out_channels=input_dim),
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias,
                                     padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def encode(self, x):
        x = x.view(-1, self.mode, self.cube_len, self.cube_len, self.cube_len)
        out = x
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = self.layer6(out)
        #print(out.shape)
        out = self.layer7(out)
        #print(out.shape)
        out = out.view(-1, self.f_dim * 64)
        return self.fc1(out), self.fc2(out)

    def decode(self, x):
        x = x.view(-1, self.z_dim, 1, 1, 1)
        out = x
        out = self.layer8(out)
        #print(out.shape)
        out = self.layer9(out)
        #print(out.shape)
        out = self.layer10(out)
        #print(out.shape)
        out = self.layer11(out)
        #print(out.shape)
        out = self.layer12(out)
        #print(out.shape)
        out = self.layer13(out)
        #print(out.shape)
        out = self.layer14(out)
        #print(out.shape)
        out = torch.squeeze(out)
        return out

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # x[256,256,6]
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
class net_D(torch.nn.Module):
    def __init__(self):
        super(net_D, self).__init__()
        self.cube_len = params.cube_len
        self.leak_value = params.leak_value
        self.bias = params.bias
        self.mode = params.mode
        self.f_dim = (int(self.mode / 4) + 1) * 4
        padd = (0,0,0)
        if self.cube_len == 128:
            padd = (1,1,1)
        self.layer1 = self.conv_layer(self.mode, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer5 = self.conv_layer(self.f_dim * 8, self.f_dim * 16, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias)
        self.layer6 = self.conv_layer(self.f_dim * 16, self.f_dim * 32, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias)
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim*32, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        out = x.view(-1, self.mode, self.cube_len, self.cube_len, self.cube_len)
        # print(out.size()) # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        # out = out.view(-1, 256*2*2*2)
        # print (out.size())
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        # print(out.size())  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out
