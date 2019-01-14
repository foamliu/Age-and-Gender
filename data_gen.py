import json
import pickle
import random

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from utils import align_face
from config import *


class AgeGenDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        samples = data['samples']

        num_samples = len(samples)
        num_train = int(train_split * num_samples)

        if split == 'train':
            samples = samples[:num_train]
        else:
            samples = samples[num_train:]

        self.samples = samples

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        # img = cv.imread(full_path)
        img = align_face(full_path, landmarks)
        # loc = sample['face_location']
        # x1, y1, x2, y2 = loc[0], loc[1], loc[2], loc[3]
        # img = img[y1:y2, x1:x2]
        # img = cv.resize(img, (image_w, image_h))
        # print('img.shape: ' + str(img.shape))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, image_h, image_w)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        age = sample['age']
        gender = sample['gender']
        return img, age, gender

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        np.random.shuffle(self.samples)


if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = random.sample(data['samples'], 10)

    sample_inputs = []
    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        age = sample['age']
        gender = sample['gender']
        face_location = sample['face_location']
        x1, y1, x2, y2 = face_location[0], face_location[1], face_location[2], face_location[3]
        print(gender, age, full_path)
        img = cv.imread(full_path)
        img = img[y1:y2, x1:x2]
        img = cv.resize(img, (image_w, image_h))
        filename = 'images/{}_img.jpg'.format(i)
        cv.imwrite(filename, img)
        sample_inputs.append({'i': i, 'gender': gender, 'age': age})

    with open('sample_inputs.json', 'w') as file:
        json.dump(sample_inputs, file, indent=4, ensure_ascii=False)
