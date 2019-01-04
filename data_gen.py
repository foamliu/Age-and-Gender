import cv2 as cv
import numpy as np
import scipy.io
from torch.utils.data import Dataset

from config import *
from utils import get_sample


class AGDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split):
        mat = scipy.io.loadmat('data/imdb/imdb.mat')
        imdb = mat['imdb'][0][0]
        num_samples = len(imdb[0][0])

        samples = []

        for i in range(num_samples):
            sample = get_sample(imdb, i)
            samples.append(sample)

        self.samples = samples

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        sample = self.samples[i]
        full_path = sample['full_path']
        # Read images
        img = cv.imread(full_path)
        img = cv.resize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)

        age = sample['age']
        gender = sample['gender']

        return img, age, gender

    def __len__(self):
        return len(self.samples)
