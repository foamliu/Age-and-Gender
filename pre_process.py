import os
import tarfile

import scipy.io
from datetime import datetime, timedelta


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


def check(mat):
    imdb = mat['imdb'][0][0]
    dob_list = imdb[0][0]
    print(dob_list)
    print(dob_list[0])
    photo_taken_list = imdb[1][0]
    print(photo_taken_list)
    print(photo_taken_list[0])
    full_path_list = imdb[2][0]
    print(full_path_list)
    print(full_path_list[0][0])
    gender_list = imdb[3][0]
    print(gender_list)
    print(gender_list[0])
    face_location_list = imdb[5][0]
    print(face_location_list)
    print(face_location_list[0][0])


if __name__ == "__main__":
    if not os.path.isdir('data/imdb_crop'):
        extract('data/imdb_crop.tar')
    if not os.path.isdir('data/imdb'):
        extract('data/imdb_meta.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
    check(mat)
