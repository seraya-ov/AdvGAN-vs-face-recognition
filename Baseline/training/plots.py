import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import functional as F


class Plotter(object):
    def __init__(self, generator_model, resnet_model, device):
        self.generator_model = generator_model
        self.attack_model = resnet_model
        self.noise = torch.randn(4 * 4, self.generator_model.noise_dim, 1, 1, requires_grad=True, device=device)
        self.device = device

    def plot_generator_results(self, test_images):
        self.generator_model.eval()
        inputs = test_images
        noise = self.generator_model(inputs)
        true_labels = np.argmax(F.softmax(self.attack_model(inputs), -1).cpu().detach().numpy(), axis=1)
        fake_labels = np.argmax(F.softmax(self.attack_model(inputs + noise), -1).cpu().detach().numpy(), axis=1)
        num = test_images.shape[0]**0.5

        plt.figure(figsize=(10, 10))
        for i in range(noise.shape[0]):
            plt.subplot(num, num, i + 1)
            plt.imshow(np.clip((inputs[i] + noise[i]).permute(1, 2, 0).cpu().detach().numpy(), 0, 1))
            plt.title("{}, {}".format(true_labels[i], fake_labels[i]))
            plt.axis('off')
        plt.show()
