import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for 4-band satellite imagery → N-class segmentation.
    """

    def __init__(self, in_channels=4, num_classes=4, base_features=32):
        super().__init__()
        f = base_features

        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f,     f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4    = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4   = DoubleConv(f * 16, f * 8)
        self.up3    = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3   = DoubleConv(f * 8,  f * 4)
        self.up2    = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2   = DoubleConv(f * 4,  f * 2)
        self.up1    = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1   = DoubleConv(f * 2,  f)

        self.out_conv = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class ProductionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])