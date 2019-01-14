import pickle
import random
import numpy as np
from config import *
from align_faces import align_face

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint_.pth.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    num_samples = len(samples)
    num_train = int(train_split * num_samples)
    samples = samples[num_train:]

    samples = random.sample(samples, 10)

    for sample in samples:
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        img = align_face(full_path, landmarks)
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, image_h, image_w)
        assert np.max(img) <= 255
        inputs = torch.FloatTensor(img / 255.)
        age_out, gen_out = model(inputs)
