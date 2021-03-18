import torch
import numpy as np


class Trainer(object):
    def __init__(self, generator,
                 discriminator,
                 attacked_model,
                 gan_loss,
                 attack_loss,
                 hinge_loss,
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
        self.hinge_loss = hinge_loss

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

        self.checkpoint_path = checkpoint_path

    def train_one_epoch(self, batch, labels=None):
        self.generator.to(device=self.device).train()
        self.discriminator.to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)

        if labels is None:
            labels = self.attacked_model(batch).argmax(dim=-1)

        labels = labels.to(device=self.device)

        discriminator_real = self.discriminator(batch)
        discriminator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        self.discriminator_optimizer.zero_grad()

        disc_loss = self.gan_loss.discriminator_loss(discriminator_real,
                                                     discriminator_fake).item()

        self.discriminator_optimizer.step()

        generator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        self.generator_optimizer.zero_grad()

        gen_loss = self.gan_loss.generator_loss(generator_fake).item()
        attack_loss = self.attack_loss.loss((batch + self.generator(batch)) / 2, labels).item()
        hinge_loss = self.hinge_loss.loss(self.generator(batch)).item()

        self.generator_optimizer.step()

        return gen_loss, disc_loss, attack_loss, hinge_loss

    def train(self, train_data, epochs):
        generator_loss = []
        discriminator_loss = []
        model_loss = []
        hinge_loss = []
        for epoch in range(epochs):
            generator_loss_epoch = []
            discriminator_loss_epoch = []
            model_loss_epoch = []
            hinge_loss_epoch = []
            for i, batch in enumerate(train_data):
                loss = self.train_one_epoch(batch[0])
                generator_loss_epoch.append(loss[0])
                discriminator_loss_epoch.append(loss[1])
                model_loss_epoch.append(loss[2])
                hinge_loss_epoch.append(loss[3])

            generator_loss.append(np.array(generator_loss_epoch).mean())
            discriminator_loss.append(np.array(discriminator_loss_epoch).mean())
            model_loss.append(np.array(model_loss_epoch).mean())
            hinge_loss.append(np.array(hinge_loss_epoch).mean())

            print("Epoch: {},"
                  " Generator loss: {},"
                  " Discriminator loss: {},"
                  " Model loss: {},"
                  " Hinge loss: {},".format(epoch,
                                            np.array(
                                                generator_loss_epoch).mean(),
                                            np.array(
                                                discriminator_loss_epoch).mean(),
                                            np.array(
                                                model_loss_epoch).mean(),
                                            np.array(
                                                hinge_loss_epoch).mean(),
                                            ))

        return generator_loss, discriminator_loss, model_loss, hinge_loss


class BaseClassifierTrainer(object):
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, batch, real_labels):
        self.model.to(device=self.device).train()
        batch = batch.to(device=self.device, dtype=torch.float32)

        labels = self.model(batch)

        self.optimizer.zero_grad()

        loss = self.criterion(labels, real_labels)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train(self, train_data, epochs):
        loss = []
        for epoch in range(epochs):
            loss_epoch = []
            for i, batch in enumerate(train_data):
                step_loss = self.train_one_epoch(*batch)
                loss_epoch.append(step_loss)

            loss.append(np.array(loss_epoch).mean())

            print("Epoch: {},"
                  " Loss: {}".format(epoch,
                                     np.array(loss_epoch).mean()))

        return loss
