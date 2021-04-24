import torch
from torch import nn, autograd


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

    def generator_loss(self, fake_output, _=None):
        fake = self.criterion(fake_output.squeeze(), torch.full((self.batch_size,),
                                                                1, dtype=torch.float,
                                                                device=self.device).squeeze())
        fake.backward()
        return fake.mean()


class WGANLoss(object):
    def __init__(self, batch_size, device, criterion=nn.BCELoss(), penalty=0.5):
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.gradient_penalty = penalty

    def discriminator_loss(self, real_output, fake_output, penalty):
        loss = (real_output.mean(dim=0) - fake_output.mean(dim=0) + self.gradient_penalty * penalty).mean()
        loss.backward()
        return loss.mean()

    @staticmethod
    def generator_loss(fake_output, _=None):
        fake = -fake_output.squeeze().mean()
        fake.backward()
        return fake.mean()


class DiscriminatorLoss(nn.Module):
    def __init__(self, batch_size, device, criterion=nn.BCELoss()):
        super().__init__()
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device

    def forward(self, real_output, fake_output):
        real = self.criterion(real_output.squeeze(), torch.full((self.batch_size,),
                                                                1, dtype=torch.float,
                                                                device=self.device).squeeze())
        fake = self.criterion(fake_output.squeeze(), torch.full((self.batch_size,),
                                                                0, dtype=torch.float,
                                                                device=self.device).squeeze())
        return real + fake


class GeneratorLoss(nn.Module):
    def __init__(self, batch_size, device, criterion=nn.BCELoss()):
        super().__init__()
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device

    def forward(self, fake_output, _=None):
        fake = self.criterion(fake_output.squeeze(), torch.full((self.batch_size,),
                                                                1, dtype=torch.float,
                                                                device=self.device).squeeze())
        return fake


class AttackLoss(nn.Module):
    def __init__(self, alpha=0.5, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, model_output, labels):
        loss = -self.alpha * self.criterion(model_output, labels.squeeze())
        return loss


class CustomAttackLoss(object):
    def __init__(self, alpha=0.5, criterion=nn.CrossEntropyLoss()):
        self.criterion = criterion
        self.alpha = alpha

    def loss(self, model_output, labels):
        loss = -self.alpha * self.criterion(model_output, labels.squeeze())
        loss.backward()
        return loss.mean()


class HingeLoss(nn.Module):
    def __init__(self, batch_size, device, alpha=0.5, criterion=nn.HingeEmbeddingLoss(margin=-1)):
        super().__init__()
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

    def forward(self, generated_output, _=None):
        loss = self.alpha * self.criterion(-torch.norm(generated_output, dim=[-1, -2, -3], p=2),
                                           torch.full((self.batch_size,),
                                                      -1, dtype=torch.long,
                                                      device=self.device).squeeze())
        return loss


class CustomHingeLoss(object):
    def __init__(self, batch_size, device, alpha=0.5, criterion=nn.HingeEmbeddingLoss(margin=-1)):
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

    def loss(self, generated_output, _=None):
        loss = self.alpha * self.criterion(-torch.norm(generated_output, dim=[-1, -2, -3], p=2),
                                           torch.full((self.batch_size,),
                                                      -1, dtype=torch.long,
                                                      device=self.device).squeeze())
        loss.backward()
        return loss.mean()


class CustomWGANDiscriminatorLoss(object):
    def __init__(self, gradient_penalty, device):
        self.device = device
        self.gradient_penalty = gradient_penalty

    def loss(self, output, gradient_penalty):
        real_output, fake_output = output
        loss = (real_output.mean(dim=0) - fake_output.mean(dim=0) + self.gradient_penalty * gradient_penalty).mean()
        loss.backward()
        return loss


class CustomWGANGeneratorLoss(object):
    @staticmethod
    def loss(fake_output, _=None):
        fake = -fake_output.squeeze().mean()
        fake.backward()
        return fake


class WGANDiscriminatorLoss(nn.Module):
    def __init__(self, gradient_penalty, device):
        super().__init__()
        self.device = device
        self.gradient_penalty = gradient_penalty

    def forward(self, output, gradient_penalty):
        real_output, fake_output = output
        loss = (real_output.mean(dim=0) - fake_output.mean(dim=0) + self.gradient_penalty * gradient_penalty).mean()
        return loss


class WGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(fake_output, _=None):
        fake = -fake_output.squeeze().mean()
        return fake
