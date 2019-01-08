import datetime
import os
import pickle
import tarfile

import cv2 as cv
import numpy as np
import scipy.io
import seaborn as sns

sns.set(color_codes=True)
from collections import Counter
from tqdm import tqdm

from config import IMG_DIR, DATA_DIR, pickle_file
from utils import get_sample


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def reformat_date(mat_date):
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt


def create_path(path):
    return os.path.join(IMG_DIR, path[0])


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
        age = sample['age']
        full_path = sample['full_path']
        gender = sample['gender']
        face_location = sample['face_location']
        x1 = int(round(face_location[0]))
        y1 = int(round(face_location[1]))
        x2 = int(round(face_location[2]))
        y2 = int(round(face_location[3]))
        print(full_path)
        print(x1, y1, x2, y2)
        img = cv.imread(full_path)
        h, w = img.shape[:2]
        print(h, w)
        samples.append({'age': age, 'gender': gender})
        filename = os.path.join('data/temp', str(i) + '.jpg')
        new_img = img[y1:y2, x1:x2, :]
        cv.imwrite(filename, new_img)


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    imdb = mat['imdb'][0, 0]
    data = [d[0] for d in imdb]
    keys = ['dob',
            'photo_taken',
            'full_path',
            'gender',
            'name',
            'face_location',
            'face_score',
            'second_face_score',
            'celeb_names',
            'celeb_id'
            ]
    imdb_dict = dict(zip(keys, np.asarray(data)))
    imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
    imdb_dict['full_path'] = [create_path(path) for path in imdb_dict['full_path']]

    # Add 'age' key to the dictionary
    imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']

    print("Dictionary created...")

    raw_path = imdb_dict['full_path']
    raw_age = imdb_dict['age']
    raw_gender = imdb_dict['gender']
    raw_sface = imdb_dict['second_face_score']

    age = []
    gender = []
    imgs = []
    samples = []
    current_age = np.zeros(101)
    for i, sface in enumerate(raw_sface):
        if np.isnan(sface) and raw_age[i] >= 0 and raw_age[i] <= 100 and not np.isnan(raw_gender[i]):
            age_tmp = 0
            if current_age[raw_age[i]] >= 5000:
                continue
            age.append(raw_age[i])
            gender.append(raw_gender[i])
            imgs.append(raw_path[i])
            samples.append({'age': raw_age[i], 'gender': raw_gender[i], 'full_path': raw_path[i]})
            current_age[raw_age[i]] += 1

    sns.distplot(age)
    print("Age size: " + str(len(age)))

    counter = Counter(age)
    print(counter)

    try:
        f = open(pickle_file, 'wb')
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    # num_samples = len(imdb[0][0])
    # print('num_samples: ' + str(num_samples))
    # check_one(imdb, 10)
    # check(imdb, num_samples)
