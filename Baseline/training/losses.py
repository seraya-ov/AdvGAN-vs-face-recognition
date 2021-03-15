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
    def __init__(self, model, batch_size, device, alpha=0.5, criterion=nn.CrossEntropyLoss()):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

    def loss(self, fake_output, labels):
        loss = -self.alpha * self.criterion(self.model(fake_output), labels.squeeze())
        loss.backward()
        return loss.mean()


class HingeLoss(object):
    def __init__(self, model, batch_size, device, alpha=0.5, criterion=nn.HingeEmbeddingLoss()):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

    def loss(self, generated_output):
        loss = self.alpha * self.criterion(torch.norm(generated_output.squeeze(), dim=[-1, -2]),
                                           torch.full((self.batch_size,),
                                                      1, dtype=torch.long,
                                                      device=self.device).squeeze())
        loss.backward()
        return loss.mean()
