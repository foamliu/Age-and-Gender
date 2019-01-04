import os
from datetime import datetime, timedelta

from config import *


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


def get_sample(imdb, i):
    sample = dict()
    dob_list = imdb[0][0]
    dob = dob_list[i]
    dob = convert_matlab_datenum(dob)
    sample['dob'] = dob
    photo_taken_list = imdb[1][0]
    photo_taken = int(photo_taken_list[i])
    sample['photo_taken'] = photo_taken
    age = num_years(dob, photo_taken)
    sample['age'] = age
    full_path_list = imdb[2][0]
    full_path = os.path.join(image_folder, full_path_list[i][0])
    sample['full_path'] = full_path
    gender_list = imdb[3][0]
    gender = int(gender_list[i])
    sample['gender'] = gender
    face_location_list = imdb[5][0]
    face_location = face_location_list[i][0]
    sample['face_location'] = face_location
    return sample
