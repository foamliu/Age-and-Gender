import os
import tarfile

import cv2 as cv
import scipy.io
from tqdm import tqdm

from utils import get_sample


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def check_one(imdb, i=10):
    sample = get_sample(imdb, i)
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


def check(imdb, num_samples):
    samples = []
    for i in tqdm(range(num_samples)):
        sample = get_sample(imdb, i)
        dob = sample['dob']
        photo_taken = sample['photo_taken']
        age = sample['age']
        full_path = sample['full_path']
        gender = sample['gender']
        face_location = sample['face_location']
        x1 = int(face_location[0])
        y1 = int(face_location[1])
        x2 = int(face_location[2])
        y2 = int(face_location[3])
        img = cv.imread(full_path)
        samples.append({'age': age, 'gender': gender})
        filename = os.path.join('data/temp', str(i) + '.jpg')
        new_img = img[y1:y2, x1:x2]
        cv.imwrite(filename, new_img)


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    imdb = mat['imdb'][0][0]
    num_samples = len(imdb[0][0])
    print('num_samples: ' + str(num_samples))
    check_one(imdb, 10)
    check(imdb, num_samples)
