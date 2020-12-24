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


class AttackLoss(object):
    def __init__(self, resnet, batch_size, device, target_class=0, criterion=nn.CrossEntropyLoss()):
        self.model = resnet
        self.criterion = criterion
        self.target_class = target_class
        self.device = device
        self.batch_size = batch_size

    def loss(self, fake_output):
        loss = self.criterion(self.model(fake_output), torch.full((self.batch_size,),
                                                                  self.target_class, dtype=torch.long,
                                                                  device=self.device).squeeze())
        loss.backward()
        return loss.mean()
