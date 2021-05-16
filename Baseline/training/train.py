import os
import shutil

from PIL import Image
from tqdm.notebook import tqdm

from skimage import transform

import numpy as np
import pandas as pd
from .losses import *


class CustomTrainer(object):
    def __init__(self,
                 model: dict,
                 optimizer: dict,
                 scheduler: dict,
                 loss: dict,
                 device: str,
                 clean=True,
                 start_epoch=0,
                 log_path=None,
                 image_path=None,
                 penalty=True):
        """
        Training loop
            :param model: generator, discriminator & attacked_model models
            :param optimizer: generator and discriminator optimizers
            :param scheduler: generator and discriminator schedulers
            :param loss: generator, discriminator, hinge & attack loss functions
            :param device: device
            :param clean: delete log dir
            :param start_epoch: epoch to start with
            :param log_path: log dir
            :param image_path: path to test image
            :param penalty: use gradient penalty
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.log_path = log_path
        self.start_epoch = start_epoch

        if log_path is not None and clean:
            try:
                shutil.rmtree(self.log_path)
            except Exception as e:
                print(e)

            os.mkdir(self.log_path)
            os.mkdir(os.path.join(self.log_path, 'images'))
            os.mkdir(os.path.join(self.log_path, 'models'))

        self.image_path = image_path

        self.device = device

        self.penalty = penalty

    def gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.Tensor(np.random.random((real_samples.shape[0], 1, 1, 1))).to(device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.model['discriminator'](interpolates)
        fake = torch.autograd.Variable(torch.Tensor(real_samples.shape[0], 1).to(device=self.device).fill_(1.0),
                                       requires_grad=False)

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
        self.model['generator'].to(device=self.device).train()
        self.model['discriminator'].to(device=self.device).train()

        batch = batch.to(device=self.device, dtype=torch.float32)

        if labels is None:
            labels = self.model['attacked_model'](batch).argmax(dim=-1)

        labels = labels.to(device=self.device)

        discriminator_real = self.model['discriminator'](batch)
        discriminator_fake = self.model['discriminator'](self.model['generator'](batch)[0])

        self.optimizer['discriminator'].zero_grad()

        if self.penalty:
            gradient_penalty = self.gradient_penalty(batch, self.model['generator'](batch)[0])
            disc_loss = self.loss['discriminator']((discriminator_real, discriminator_fake), gradient_penalty)
            disc_loss.backward()
        else:
            disc_loss = self.loss['discriminator'](discriminator_real, discriminator_fake)
            disc_loss.backward()

        self.optimizer['discriminator'].step()
        self.scheduler['discriminator'].step(disc_loss)

        generator_fake = self.model['discriminator'](self.model['generator'](batch)[0])

        self.optimizer['generator'].zero_grad()

        gen_loss = self.loss['generator'](generator_fake)
        gen_loss.backward()

        attack_loss = self.loss['attack'](self.model['attacked_model'](self.model['generator'](batch)[0]), labels)
        attack_loss.backward()

        hinge_loss = self.loss['hinge'](self.model['generator'](batch)[1])
        hinge_loss.backward()

        self.optimizer['generator'].step()
        self.scheduler['generator'].step()

        return {'generator_loss': gen_loss.item(),
                'discriminator_loss': disc_loss.item(),
                'attack_loss': attack_loss.item(),
                'hinge_loss': hinge_loss.item()
                }

    def log_metrics(self, metrics: dict, progress_bar):
        if self.log_path is not None:
            metrics_path = os.path.join(self.log_path, 'metrics.csv')
            try:
                past_metrics = pd.read_csv(metrics_path)
                pd.DataFrame(pd.concat([past_metrics, pd.DataFrame(metrics, index=[0])],
                                       ignore_index=True)).to_csv(metrics_path, index=False)
            except Exception:
                pd.DataFrame(metrics, index=[0]).to_csv(metrics_path, index=False)

        progress_bar.set_description(str(metrics))

    def log_image(self, image_size, epoch):
        if self.image_path is not None:
            image = torch.tensor(transform.resize(np.array(Image.open(self.image_path)), (image_size, image_size)),
                                 device=self.device, dtype=torch.float)
            if len(image.shape) > 2:
                image = image.permute((2, 0, 1))
            else:
                image = image.unsqueeze(0)

            image = image.unsqueeze(0)
            self.model['generator'].eval()
            generated = self.model['generator'](image)[0].squeeze()
            self.model['generator'].train()

            if len(generated.shape) > 2:
                generated = generated.permute((1, 2, 0))

            generated = generated.cpu().detach().numpy()

            Image.fromarray((generated * 255).astype(np.uint8)).save(os.path.join(self.log_path,
                                                                                  'images/{}.jpg'.format(epoch)))

    def log_model(self, epoch):
        torch.save(self.model['generator'].state_dict(),
                   os.path.join(self.log_path, 'models/gen_checkpoint_{}'.format(epoch)))
        torch.save(self.model['discriminator'].state_dict(),
                   os.path.join(self.log_path, 'models/disc_checkpoint_{}'.format(epoch)))

    def train(self, train_data, epochs):
        """
        Train model
        :param train_data: dataset
        :param epochs: number of epochs to train for
        """
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            self.log_image(self.model['generator'].image_size, epoch)
            self.log_model(epoch)
            progress_bar = tqdm(enumerate(train_data))
            for i, batch in progress_bar:
                metrics = self.train_one_epoch(batch[0])
                metrics['epoch'] = epoch
                metrics['batch'] = i
                self.log_metrics(metrics, progress_bar)

        return self
