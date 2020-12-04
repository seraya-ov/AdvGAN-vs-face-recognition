import torch
import numpy as np
from IPython.display import clear_output
from .plots import Plotter


class Trainer(object):
    def __init__(self, generator,
                 discriminator,
                 loss,
                 discriminator_optimizer,
                 generator_optimizer,
                 device,
                 checkpoint_path=''):

        self.generator = generator.to(device)
        self.noise_dim = generator.noise_dim

        self.discriminator = discriminator.to(device)

        self.plotter = Plotter(generator, device)

        self.device = device
        self.loss = loss

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.checkpoint_path = checkpoint_path

    def noise(self, batch_size, input_size):
        return torch.randn(batch_size, input_size, 1, 1, requires_grad=True, device=self.device)

    def train_one_epoch(self, batch):
        self.generator.to(device=self.device).train()
        self.discriminator.to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)
        noise = self.noise(batch.shape[0], self.noise_dim).to(device=self.device, dtype=torch.float32)

        discriminator_real = self.discriminator(batch)
        generator_output = self.generator(noise)

        discriminator_fake = self.discriminator(generator_output.detach())

        self.discriminator_optimizer.zero_grad()
        discriminator_loss = self.loss.discriminator_loss(discriminator_real,
                                                          discriminator_fake)
        self.discriminator_optimizer.step()

        disc_loss = discriminator_loss.item()

        generator_fake = self.discriminator(generator_output)

        self.generator_optimizer.zero_grad()
        generator_loss = self.loss.generator_loss(generator_fake)
        self.generator_optimizer.step()
        gen_loss = generator_loss.item()

        return gen_loss, disc_loss

    def train(self, train_data, epochs):
        generator_loss = []
        discriminator_loss = []
        for epoch in range(epochs):
            generator_loss_epoch = []
            discriminator_loss_epoch = []
            for i, batch in enumerate(train_data):
                loss = self.train_one_epoch(batch[0])
                generator_loss_epoch.append(loss[0])
                discriminator_loss_epoch.append(loss[1])
                if i % 50 == 0:
                    clear_output(wait=True)
                    self.plotter.plot_generator_results(random=False)

            generator_loss.append(np.array(generator_loss_epoch).mean())
            discriminator_loss.append(np.array(discriminator_loss_epoch).mean())

            print("Epoch: {}, Generator loss: {}, Discriminator loss: {}".format(epoch,
                                                                                 np.array(generator_loss_epoch).mean(),
                                                                                 np.array(
                                                                                     discriminator_loss_epoch).mean()))
            clear_output(wait=True)
            self.plotter.plot_generator_results(random=False)

        return generator_loss, discriminator_loss
