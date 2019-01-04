import os
import tarfile
from datetime import datetime, timedelta

import scipy.io

from config import *


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def convert_matlab_datenum(matlab_datenum):
    part_1 = datetime.fromordinal(int(matlab_datenum))
    part_2 = timedelta(days=int(matlab_datenum) % 1)
    part_3 = timedelta(days=366)
    python_datetime = part_1 + part_2 - part_3
    return python_datetime


def num_years(begin, end_year):
    end = datetime(end_year, 6, 30)
    num_years = int((end - begin).days / 365.2425)
    return num_years


def check(mat):
    i = 10
    num_samples = len(mat['imdb'][0][0][0][0])
    print('num_samples: ' + str(num_samples))
    imdb = mat['imdb'][0][0]
    dob_list = imdb[0][0]
    print(dob_list)
    dob = dob_list[i]
    print('dob: ' + str(dob))
    dob = convert_matlab_datenum(dob)
    print('dob: ' + str(dob))
    photo_taken_list = imdb[1][0]
    print(photo_taken_list)
    photo_taken = int(photo_taken_list[i])
    print('photo_taken: ' + str(photo_taken))
    age = num_years(dob, photo_taken)
    print('age: ' + str(age))
    full_path_list = imdb[2][0]
    print(full_path_list)
    full_path = os.path.join(image_folder, full_path_list[i][0])
    print('full_path: ' + str(full_path))
    gender_list = imdb[3][0]
    print(gender_list)
    gender = int(gender_list[i])
    print('gender: ' + str(gender))
    face_location_list = imdb[5][0]
    print(face_location_list)
    face_location = face_location_list[i][0]
    print('face_location: ' + str(face_location))


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    check(mat)
