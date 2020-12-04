from torch import nn


class ConvBnReLuBlock(nn.Module):
    def __init__(self, in_channels=1, hidden_size=32, kernel_size=4, stride=1, padding=0, transpose=True):
        super(ConvBnReLuBlock, self).__init__()
        if transpose:
            conv = nn.ConvTranspose2d(in_channels, hidden_size, kernel_size, stride, padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels, hidden_size, kernel_size, stride, padding, bias=False)
        self.model = nn.Sequential(
            conv,
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),
        )

    def forward(self, batch):
        return self.model(batch)


class Generator(nn.Module):
    def __init__(self, in_channels=100, hidden_size=32, out_channels=1):
        super(Generator, self).__init__()
        self.noise_dim = in_channels
        self.model = nn.Sequential(

            ConvBnReLuBlock(in_channels, hidden_size * 2, 3, 1, 0),
            ConvBnReLuBlock(hidden_size * 2, hidden_size, 3, 2, 1),
            # ConvBnReLuBlock(hidden_size * 2, hidden_size, 4, 2, 1),
            # ConvBnReLuBlock(hidden_size * 2, hidden_size, 4, 2, 1),

            nn.ConvTranspose2d(hidden_size, out_channels, 3, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, batch):
        return self.model(batch)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, hidden_size=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(

            ConvBnReLuBlock(in_channels, hidden_size, 3, 2, 1, transpose=False),
            ConvBnReLuBlock(hidden_size, hidden_size * 2, 3, 2, 1, transpose=False),
            # ConvBnReLuBlock(hidden_size * 2, hidden_size * 4, 4, 2, 1),
            # ConvBnReLuBlock(hidden_size * 4, hidden_size * 8, 4, 2, 1),

            nn.Conv2d(hidden_size * 2, 1, 3, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, batch):
        return self.model(batch)
