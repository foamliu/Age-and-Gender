from torch.utils.data import Dataset
from datetime import datetime, timedelta
import os
import numpy as np
import scipy.io
from config import *


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, transform=None):
        mat = scipy.io.loadmat('data/imdb/imdb.mat')
        imdb = mat['imdb'][0][0]
        num_samples = len(imdb[0][0])

        samples = []

        for i in range(num_samples):
            


        self.num_samples = num_samples

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        sample = self.samples[i]
        path = os.path.join(image_folder, sample['image_id'])
        # Read images
        img = imread(path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        # Sample captions
        captions = sample['caption']
        # Sanity check
        assert len(captions) == captions_per_image
        c = captions[i % captions_per_image]
        c = list(jieba.cut(c))
        # Encode captions
        enc_c = encode_caption(self.word_map, c)

        caption = torch.LongTensor(enc_c)

        caplen = torch.LongTensor([len(c) + 2])

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor([encode_caption(self.word_map, list(jieba.cut(c))) for c in captions])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.num_samples


