from .gan import *


class SimpleClassifier(nn.Module):
    def __init__(self, in_channels=1, image_size=10, hidden_size=32, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.model = nn.Sequential(

            ConvBnReLuBlock(in_channels, hidden_size, 3, 1, 1, transpose=False),
            ConvBnReLuBlock(hidden_size, hidden_size * 2, 3, 1, 1, transpose=False),
            ConvBnReLuBlock(hidden_size * 2, hidden_size * 4, 3, 1, 1),
            # ConvBnReLuBlock(hidden_size * 4, hidden_size * 8, 3, 1, 1),

            ConvBnReLuBlock(hidden_size * 4, 1, 3, 1, 1, transpose=False),
            nn.Flatten(),
            nn.Linear(image_size * image_size, num_classes),
        )

    def forward(self, batch):
        return self.model(batch)
