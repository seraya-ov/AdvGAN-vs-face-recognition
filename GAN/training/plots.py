import matplotlib.pyplot as plt
import torch


class Plotter(object):
    def __init__(self, generator_model, device):
        self.generator_model = generator_model
        self.noise = torch.randn(4 * 4, self.generator_model.noise_dim, 1, 1, requires_grad=True, device=device)
        self.device = device

    def plot_generator_results(self, num=4, random=True):
        self.generator_model.eval()
        if random:
            inputs = torch.randn(num * num, self.generator_model.noise_dim, 1, 1, requires_grad=True, device=self.device)
        else:
            inputs = self.noise
        images = self.generator_model(inputs).permute((0, 2, 3, 1)).cpu().detach().numpy()

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(num, num, i + 1)
            plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.show()
