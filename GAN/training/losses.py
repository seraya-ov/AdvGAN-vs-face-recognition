import torch
from torch import nn


class GANLoss(object):
    def __init__(self, batch_size, device, criterion=nn.BCELoss()):
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device

    def discriminator_loss(self, real_output, fake_output):
        real = self.criterion(real_output.squeeze(), torch.full((self.batch_size,),
                                                                1, dtype=torch.float,
                                                                device=self.device).squeeze())
        real.backward()
        fake = self.criterion(fake_output.squeeze(), torch.full((self.batch_size,),
                                                                0, dtype=torch.float,
                                                                device=self.device).squeeze())
        fake.backward()
        return (real + fake).mean()

    def generator_loss(self, fake_output):
        fake = self.criterion(fake_output.squeeze(), torch.full((self.batch_size,),
                                                                1, dtype=torch.float,
                                                                device=self.device).squeeze())
        fake.backward()
        return fake.mean()
