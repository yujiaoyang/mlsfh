# import torchvision
import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, *channels, p=0.4, transposed=False):
        super(BasicConv, self).__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=2, bias=False) \
                        if not transposed else \
                        nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=2, bias=False),
                    nn.BatchNorm2d(num_features=c_out),
                    nn.ELU(inplace=True),
                    nn.Dropout(p=p)
                ]
            )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.encoder = nn.Sequential(
            BasicConv(1, 32),
            BasicConv(32, 64),
            BasicConv(64, 128),
            BasicConv(128, 256),
            BasicConv(256, 512),
        )

        self.decoder = nn.Sequential(
            BasicConv(512, 256, transposed=True),
            BasicConv(256, 128, transposed=True),
            BasicConv(128, 64, transposed=True),
            BasicConv(64, 32, transposed=True),
            BasicConv(32, 1, transposed=True)
        )

        # self.fc = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.Tanh()
        # )
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.unpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def encode(self, cmd):
        return self.encoder(cmd)

    def decode(self, cmd_enc):
        return self.decoder(cmd_enc)

    def forward(self, cmd):
        enc = self.encoder(cmd)
        dec = self.decoder(enc)
        return enc, dec


cmd = torch.rand(size=(10, 1, 511, 511))
sfh = FCN()
print(cmd.shape)
enc, rec = sfh(cmd)
print(enc.shape, rec.shape)

"""
Hout = (Hin+2*0-1*(kernel_size-1)-1)/stride + 1
Hout = (Hin-3)/2 + 1
Hin = (Hout-1)*2+3
     = 1
Hin = 3
"""
x = 1
for i in range(10):
    print(x)
    x = (x - 1) * 2 + 3
