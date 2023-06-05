import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, input, output):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=True), 
            nn.BatchNorm2d(output), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=True), 
            nn.BatchNorm2d(output), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, input, output):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=True), 
		    nn.BatchNorm2d(output), 
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    def __init__(self, input=3, output=1):
        super(UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(input=input, output=64)
        self.conv2 = conv_block(input=64, output=128)
        self.conv3 = conv_block(input=128, output=256)
        self.conv4 = conv_block(input=256, output=512)
        self.conv5 = conv_block(input=512, output=1024)

        self.up5 = up_conv(input=1024, output=512)
        self.up_conv5 = conv_block(input=1024, output=512)

        self.up4 = up_conv(input=512, output=256)
        self.up_conv4 = conv_block(input=512, output=256)
        
        self.up3 = up_conv(input=256, output=128)
        self.up_conv3 = conv_block(input=256, output=128)
        
        self.up2 = up_conv(input=128, output=64)
        self.up_conv2 = conv_block(input=128, output=64)

        self.conv_1x1 = nn.Conv2d(64, output, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1