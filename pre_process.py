import os
import tarfile

import scipy.io

from utils import get_sample


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def check(mat, i=10):
    imdb = mat['imdb'][0][0]
    sample = get_sample(imdb, i)

    num_samples = len(imdb[0][0])
    print('num_samples: ' + str(num_samples))

    dob = sample['dob']
    print('dob: ' + str(dob))
    photo_taken = sample['photo_taken']
    print('photo_taken: ' + str(photo_taken))
    age = sample['age']
    print('age: ' + str(age))
    full_path = sample['full_path']
    print('full_path: ' + str(full_path))
    gender = sample['gender']
    print('gender: ' + str(gender))
    face_location = sample['face_location']
    print('face_location: ' + str(face_location))


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    check(mat, 10)
