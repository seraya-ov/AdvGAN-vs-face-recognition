# def read_and_prep_images(img_paths,
# transforms=[lambda x: x], transform=False, transform_add=0, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE):
#     if transform:
#         imgs = [random.choice(transforms)
#         (image=np.array(load_img(img_path, target_size=(img_height, img_width))))['image'] for img_path in img_paths]
#     else:
#         imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
#     samples = random.sample(imgs, transform_add)
#     imgs.extend([random.choice(transforms)(image=np.array(samples[i]))['image'] for i in range(transform_add)])
#     img_array = np.array([img_to_array(img) for img in imgs])
#     output = np.array(preprocess_input(img_array))
#     return(imgs, output)

from skimage import io  # , transform
from skimage.color import gray2rgb

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import torch

# import torch.nn as nn
from torch.utils.data import Dataset
# from torchvision import transforms, utils

import os


class FacesDataset(Dataset):
    """Face Landmarks dataset."""

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
