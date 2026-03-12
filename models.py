import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard Convolutional Autoencoder
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# UNet
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
    
# ResNetAE
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv_block(x)) # Element-wise Sum (The "Residual" part)

class ResNetAE(nn.Module):
    def __init__(self):
        super(ResNetAE, self).__init__()
        # Encoder
        self.start = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
        self.res1 = ResidualBlock(64)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # Down
        self.res2 = ResidualBlock(128)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # Down
        
        # Bottleneck
        self.res3 = ResidualBlock(256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.res4 = ResidualBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.res5 = ResidualBlock(64)
        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.start(x)
        x = self.down1(self.res1(x))
        x = self.down2(self.res2(x))
        x = self.res3(x)
        x = self.res4(self.up1(x))
        x = self.final(self.res5(self.up2(x)))
        return x
    
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        
        # Flatten size depends on input (assuming 128x128 -> 16x16 feature map)
        self.fc_mu = nn.Linear(128*16*16, 512)
        self.fc_logvar = nn.Linear(128*16*16, 512)
        self.decoder_input = nn.Linear(512, 128*16*16)
        
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x_flat = x.view(x.size(0), -1)
        
        mu, logvar = self.fc_mu(x_flat), self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)
        
        z = self.decoder_input(z).view(-1, 128, 16, 16)
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        recon = torch.sigmoid(self.dec3(z))
        return recon, mu, logvar
