from abc import ABC
# from typing import Mapping, Any

# import torch
import numpy as np
import pandas as pd
from catalyst import dl
# , utils
from .losses import *
import seaborn as sns


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
                 checkpoint_path='',
                 penalty=True):

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
        self.penalty = penalty

    def gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.shape[0], 1, 1, 1))).to(device=self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).to(device=self.device).fill_(1.0),
                                       requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_one_epoch(self, batch, labels=None):
        self.generator.to(device=self.device).train()
        self.discriminator.to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)

        if labels is None:
            labels = self.attacked_model(batch).argmax(dim=-1)

        labels = labels.to(device=self.device)

        discriminator_real = self.discriminator(batch)
        discriminator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        if self.penalty:
            gradient_penalty = self.gradient_penalty(batch, discriminator_fake)

            self.discriminator_optimizer.zero_grad()

            disc_loss = self.gan_loss.discriminator_loss((discriminator_real,
                                                          discriminator_fake), gradient_penalty).item()
        else:
            self.discriminator_optimizer.zero_grad()

            disc_loss = self.gan_loss.discriminator_loss(discriminator_real,
                                                         discriminator_fake).item()

        self.discriminator_optimizer.step()

        generator_fake = self.discriminator((batch + self.generator(batch)) / 2)

        self.generator_optimizer.zero_grad()

        gen_loss = self.gan_loss.generator_loss(generator_fake).item()
        attack_loss = self.attack_loss.loss(self.attacked_model((batch + self.generator(batch)) / 2), labels).item()
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


class PlotCallback(dl.Callback):
    def __init__(self, order: int):
        super().__init__(order)
        self.logs = []

    def on_batch_end(self, runner: "GANRunner") -> None:
        log_path = runner._logdir
        metrics = pd.read_csv('./{}/train.csv'.format(log_path)).plot(figsize=(10, 5))
        mean_cols = list(filter(lambda x: 'mean' in x, metrics.columns))
        std_cols = list(filter(lambda x: 'mean' in x, metrics.columns))
        cols = list(filter(lambda x: ((x not in mean_cols) and (x not in std_cols)), metrics.columns))
        sns.lineplot(data=metrics[cols])


class GANRunner(dl.Runner, ABC):
    def __init__(self, attacked_model, batch_size, device, logdir, gradient_penalty=True):
        super().__init__()
        self._gradient_penalty = gradient_penalty
        self._device = device
        self._batch_size = batch_size
        self._attacked_model = attacked_model
        self._logdir = logdir

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def predict_batch(self, batch, **kwargs):
        images, _ = batch
        return (images + self.model['generator'](images.to(self.device)).detach()) / 2

    def get_callbacks(self, stage: str):
        if self._gradient_penalty:
            return {
                'discriminator_criterion': dl.CriterionCallback(
                    input_key='discriminator_output', target_key='gradient_penalty',
                    metric_key="discriminator loss",
                    criterion_key="discriminator_loss",
                ),
                'generator_criterion': dl.CriterionCallback(
                    input_key='generator_fake', target_key='generator_fake',
                    metric_key="generator loss",
                    criterion_key="generator_loss",
                ),
                'attack_criterion': dl.CriterionCallback(
                    input_key='attack', target_key='labels',
                    metric_key="attack loss",
                    criterion_key="attack_loss",
                ),
                'hinge_criterion': dl.CriterionCallback(
                    input_key='hinge', target_key='hinge',
                    metric_key="hinge loss",
                    criterion_key="hinge_loss",
                ),
                'generator_optimizer': dl.OptimizerCallback(
                    model_key="generator",
                    optimizer_key="generator_optimizer",
                    metric_key="generator loss"
                ),
                'discriminator_optimizer': dl.OptimizerCallback(
                    model_key="discriminator",
                    optimizer_key="discriminator_optimizer",
                    metric_key="discriminator loss"
                ),
                'checkpoint': dl.CheckpointCallback(
                    logdir=self._logdir
                ),
                "verbose": dl.TqdmCallback(),
            }
        else:
            return {
                'discriminator_criterion': dl.CriterionCallback(
                    input_key='discriminator_real', target_key='discriminator_fake',
                    metric_key="discriminator loss",
                    criterion_key="discriminator_loss",
                ),
                'generator_criterion': dl.CriterionCallback(
                    input_key='generator_fake', target_key='generator_fake',
                    metric_key="generator loss",
                    criterion_key="generator_loss",
                ),
                'attack_criterion': dl.CriterionCallback(
                    input_key='attack', target_key='labels',
                    metric_key="attack loss",
                    criterion_key="attack_loss",
                ),
                'hinge_criterion': dl.CriterionCallback(
                    input_key='hinge', target_key='hinge',
                    metric_key="hinge loss",
                    criterion_key="hinge_loss",
                ),
                'generator_optimizer': dl.OptimizerCallback(
                    model_key="generator",
                    optimizer_key="generator_optimizer",
                    metric_key="generator loss"
                ),
                'discriminator_optimizer': dl.OptimizerCallback(
                    model_key="discriminator",
                    optimizer_key="discriminator_optimizer",
                    metric_key="discriminator loss"
                ),
                'checkpoint': dl.CheckpointCallback(
                    logdir=self._logdir
                ),
                "verbose": dl.TqdmCallback(),
            }

    def gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.shape[0], 1, 1, 1))).to(device=self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.model['discriminator'](interpolates)
        fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).to(device=self.device).fill_(1.0),
                                       requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def handle_batch(self, batch):
        self.model['generator'].to(device=self.device).train()
        self.model['discriminator'].to(device=self.device).train()

        batch, labels = batch
        batch = batch.to(device=self.device, dtype=torch.float32)

        if labels is None:
            labels = self._attacked_model(batch.detach()).argmax(dim=-1)

        labels = labels.to(device=self.device)

        fake_samples = (batch + self.model['generator'](batch).detach()) / 2
        discriminator_real = self.model['discriminator'](batch)
        discriminator_fake = self.model['discriminator'](fake_samples)
        gradient_penalty = self.gradient_penalty(batch, fake_samples)

        generator_fake = self.model['discriminator']((batch + self.model['generator'](batch)) / 2)

        attack = self._attacked_model((batch + self.model['generator'](batch)) / 2)
        hinge = self.model['generator'](batch)

        self.batch = {
            "discriminator_real": discriminator_real,
            "discriminator_fake": discriminator_fake,
            "discriminator_output": (discriminator_real, discriminator_fake),
            "gradient_penalty": gradient_penalty,
            "generator_fake": generator_fake,
            "attack": attack,
            "labels": labels,
            "hinge": hinge,
        }


class CGANRunner(dl.Runner, ABC):
    def __init__(self, batch_size, device):
        super().__init__()
        self._device = device
        self._batch_size = batch_size

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def predict_batch(self, batch, **kwargs):
        images, _ = batch
        return (images + self.model['generator'](images.to(self.device)).detach()) / 2

    def handle_batch(self, batch):
        self.model['generator'].to(device=self.device).train()
        self.model['discriminator'].to(device=self.device).train()

        batch, labels = batch
        batch = batch.to(device=self.device, dtype=torch.float32)

        if labels is None:
            labels = self.model['attacked_model'](batch).argmax(dim=-1)

        labels = labels.to(device=self.device)

        discriminator_real = self.model['discriminator'](batch)
        discriminator_fake = self.model['discriminator']((batch + self.model['generator'](batch).detach()) / 2)
        discriminator_loss = self.criterion['discriminator_loss'](discriminator_real, discriminator_fake)
        discriminator_loss.backward()

        self.optimizer['discriminator_optimizer'].step()
        self.optimizer['discriminator_optimizer'].zero_grad()

        generator_fake = self.model['discriminator']((batch + self.model['generator'](batch)) / 2)
        generator_loss = self.criterion['generator_loss'](generator_fake)
        generator_loss.backward()

        attack = self.model['attacked_model']((batch + self.model['generator'](batch)) / 2)
        attack_loss = self.criterion['attack_loss'](attack, labels)
        attack_loss.backward()

        hinge = self.model['generator'](batch)
        hinge_loss = self.criterion['hinge_loss'](hinge)
        hinge_loss.backward()

        discriminator_loss = self.criterion['discriminator_loss'](discriminator_real, discriminator_fake)
        generator_loss = self.criterion['generator_loss'](generator_fake)
        attack_loss = self.criterion['attack_loss']((batch + self.model['generator'](batch)) / 2,
                                                    labels)
        hinge_loss = self.criterion['hinge_loss'](
            self.model['generator'](batch))

        self.optimizer['generator_optimizer'].step()
        self.optimizer['generator_optimizer'].zero_grad()

        metrics = {'discriminator_loss': discriminator_loss.item(),
                   'generator_loss': generator_loss.item(),
                   'attack_loss': attack_loss.item(),
                   'hinge_loss': hinge_loss.item()}

        self.batch_metrics.update(metrics)
        # for key in ['discriminator loss', 'generator loss', 'attack loss', 'hinge loss']:
        #     self.metrics[key].update(self.batch_metrics[key].item(), self.batch_size)
