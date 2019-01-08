import datetime
import os
import pickle
import tarfile

import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

from config import IMG_DIR, pickle_file
from mtcnn.detector import detect_faces


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


def is_valid_face(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    width, height = img.size
    if len(bounding_boxes) > 0:
        x1, y1, x2, y2 = bounding_boxes[0][0], bounding_boxes[0][1], bounding_boxes[0][2], bounding_boxes[0][3]
        return (x2 - x1) > width / 2 and (y2 - y1) > height / 2
    return False


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
    raw_face_loc = imdb_dict['face_location']

    age = []
    gender = []
    imgs = []
    samples = []
    current_age = np.zeros(101)
    for i, sface in tqdm(enumerate(raw_sface)):
        if np.isnan(sface) and raw_age[i] >= 0 and raw_age[i] <= 100 and not np.isnan(raw_gender[i]) and is_valid_face(
                raw_path[i]):
            age_tmp = 0
            if current_age[raw_age[i]] >= 5000:
                continue
            age.append(raw_age[i])
            gender.append(raw_gender[i])
            imgs.append(raw_path[i])
            samples.append({'age': int(raw_age[i]), 'gender': int(raw_gender[i]), 'full_path': raw_path[i]})
            current_age[raw_age[i]] += 1

    try:
        f = open(pickle_file, 'wb')
        save = {
            'age': age,
            'gender': gender,
            'samples': samples
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
