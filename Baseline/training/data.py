from skimage import io  # , transform
from skimage.color import gray2rgb

from sklearn.utils import shuffle

import torch

from torch.utils.data import Dataset
import os


class FacesDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.class_dirs = [os.path.join(self.root_dir, filename) for filename in os.listdir(self.root_dir)]
        maybe_paths = shuffle(
            [(i, os.path.join(d, filename)) for i, d in enumerate(self.class_dirs) for filename in
             os.listdir(d)])

        self.class_img_paths = []
        for cls, path in maybe_paths:
            try:
                image = io.imread(path)
                if image is not None:
                    self.class_img_paths.append((cls, path))
            except Exception:
                print(path)
                pass

        self.transform = transform

    def __len__(self):
        return len(self.class_img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_class, img_name = self.class_img_paths[idx]
        image = io.imread(img_name)
        if len(image.shape) < 3:
            image = gray2rgb(image)
        sample = {'image': image, 'class': img_class}

        if self.transform:
            sample = self.transform(sample)

        return sample
