import torch
from torch import nn
import numpy as np


# from IPython.display import clear_output
# from .plots import Plotter


class Trainer(object):
    def __init__(self, generator,
                 discriminator,
                 attacked_model,
                 gan_loss,
                 resnet_loss,
                 discriminator_optimizer,
                 generator_optimizer,
                 device,
                 checkpoint_path=''):

        self.generator = generator.to(device)
        self.noise_dim = generator.noise_dim

        self.discriminator = discriminator.to(device)
        self.attacked_model = attacked_model.to(device)

        self.plotter = None  # Plotter(generator, device)

        self.device = device
        self.gan_loss = gan_loss
        self.resnet_loss = resnet_loss

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.checkpoint_path = checkpoint_path

    def train_one_epoch(self, batch):
        self.generator.to(device=self.device).train()
        self.discriminator.to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)

        discriminator_real = self.discriminator(batch)
        discriminator_fake = self.discriminator(batch + self.generator(batch))

        self.discriminator_optimizer.zero_grad()

        discriminator_loss = self.gan_loss.discriminator_loss(discriminator_real,
                                                              discriminator_fake)
        resnet_loss = self.resnet_loss.loss(batch + self.generator(batch))

        self.discriminator_optimizer.step()

        disc_loss = discriminator_loss.item()
        resnet_loss = resnet_loss.item()

        generator_fake = self.discriminator(batch + self.generator(batch))

        self.generator_optimizer.zero_grad()

        generator_loss = self.gan_loss.generator_loss(generator_fake)
        resnet_loss += self.resnet_loss.loss(batch + self.generator(batch)).item()

        self.generator_optimizer.step()

        gen_loss = generator_loss.item()

        return gen_loss, disc_loss, resnet_loss / 2

    def train(self, train_data, epochs):
        generator_loss = []
        discriminator_loss = []
        resnet_loss = []
        for epoch in range(epochs):
            generator_loss_epoch = []
            discriminator_loss_epoch = []
            resnet_loss_epoch = []
            for i, batch in enumerate(train_data):
                loss = self.train_one_epoch(batch[0])
                generator_loss_epoch.append(loss[0])
                discriminator_loss_epoch.append(loss[1])
                resnet_loss_epoch.append(loss[2])
                # if i % 50 == 0:
                #     clear_output(wait=True)
                #     self.plotter.plot_generator_results(random=False)

            generator_loss.append(np.array(generator_loss_epoch).mean())
            discriminator_loss.append(np.array(discriminator_loss_epoch).mean())
            resnet_loss.append(np.array(resnet_loss_epoch).mean())

            print("Epoch: {},"
                  " Generator loss: {},"
                  " Discriminator loss: {},"
                  " ResNet loss: {},".format(epoch,
                                             np.array(
                                                 generator_loss_epoch).mean(),
                                             np.array(
                                                 discriminator_loss_epoch).mean(),
                                             np.array(
                                                 resnet_loss_epoch).mean(),
                                             ))

        return generator_loss, discriminator_loss, resnet_loss
