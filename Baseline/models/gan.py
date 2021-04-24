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


class Conv3Block(nn.Module):
    def __init__(self, in_channels=1, hidden_size=16):
        super(Conv3Block, self).__init__()
        self.noise_dim = in_channels
        self.model = nn.Sequential(
            ConvBnReLuBlock(in_channels, hidden_size * 4, 1, 1, 0),
            ConvBnReLuBlock(hidden_size * 4, hidden_size * 2, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 2, hidden_size, 1, 1, 0),
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size),
        )

    def forward(self, batch):
        return self.model(batch) + self.skip(batch)


class Id3Block(nn.Module):
    def __init__(self, in_channels=1, hidden_size=16):
        super(Id3Block, self).__init__()
        self.noise_dim = in_channels
        self.model = nn.Sequential(
            ConvBnReLuBlock(in_channels, hidden_size * 4, 1, 1, 0),
            ConvBnReLuBlock(hidden_size * 4, hidden_size * 2, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 2, hidden_size, 1, 1, 0),
        )

    def forward(self, batch):
        return self.model(batch) + batch


class Generator(nn.Module):
    def __init__(self, in_channels=1, hidden_size=16, out_channels=1):
        super(Generator, self).__init__()
        self.noise_dim = in_channels
        self.model = nn.Sequential(

            ConvBnReLuBlock(in_channels, hidden_size * 16, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 16, hidden_size * 8, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 8, hidden_size * 4, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 4, hidden_size * 2, 3, 1, 1),
            ConvBnReLuBlock(hidden_size * 2, hidden_size, 3, 1, 1),

            # Conv3Block(hidden_size, hidden_size),
            # Id3Block(hidden_size, hidden_size),

            nn.ConvTranspose2d(hidden_size, out_channels, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        return self.sigmoid(self.model(batch))


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, image_size=10, hidden_size=16):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(

            ConvBnReLuBlock(in_channels, hidden_size, 3, 1, 1, transpose=False),
            ConvBnReLuBlock(hidden_size, hidden_size * 2, 3, 1, 1, transpose=False),
            ConvBnReLuBlock(hidden_size * 2, hidden_size * 4, 3, 1, 1, transpose=False),
            ConvBnReLuBlock(hidden_size * 4, hidden_size * 8, 3, 1, 1, transpose=False),

            ConvBnReLuBlock(hidden_size * 8, 1, 3, 1, 1, transpose=False),
            nn.AvgPool2d(image_size),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, batch):
        return self.model(batch)
