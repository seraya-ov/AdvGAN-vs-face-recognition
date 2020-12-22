import torch
import numpy as np


class Trainer(object):
    def __init__(self, generator,
                 discriminator,
                 attacked_model,
                 gan_loss,
                 attack_loss,
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
        self.attack_loss = attack_loss

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.checkpoint_path = checkpoint_path

    def train_one_epoch(self, batch):
        self.generator.to(device=self.device).train()
        self.discriminator.to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)

        discriminator_real = self.discriminator(batch)
        discriminator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        self.discriminator_optimizer.zero_grad()

        discriminator_loss = self.gan_loss.discriminator_loss(discriminator_real,
                                                              discriminator_fake)
        attack_loss = self.attack_loss.loss((batch + self.generator(batch)) / 2)

        self.discriminator_optimizer.step()

        disc_loss = discriminator_loss.item()
        attack_loss = attack_loss.item()

        generator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        self.generator_optimizer.zero_grad()

        generator_loss = self.gan_loss.generator_loss(generator_fake)
        attack_loss += self.attack_loss.loss((batch + self.generator(batch)) / 2).item()

        self.generator_optimizer.step()

        gen_loss = generator_loss.item()

        return gen_loss, disc_loss, attack_loss / 2

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
