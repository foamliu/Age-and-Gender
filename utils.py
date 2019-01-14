import math
import os
from datetime import datetime, timedelta

import cv2 as cv
import numpy as np

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import *


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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
    try:
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
        full_path = os.path.join(IMG_DIR, full_path_list[i][0])
        sample['full_path'] = full_path
        gender_list = imdb[3][0]
        if math.isnan(gender_list[i]):
            gender = 2
        else:
            gender = int(gender_list[i])
        sample['gender'] = gender
        face_location_list = imdb[5][0]
        face_location = face_location_list[i][0]
        sample['face_location'] = face_location
        return sample
    except:
        pass
        # print('i: ' + str(i))
        # print('dob: ' + str(imdb[0][0][i]))
        # print('photo_taken: ' + str(imdb[1][0][i]))
        # print('full_path: ' + str(imdb[2][0][i][0]))
        # print('gender: ' + str(imdb[3][0][i]))
        # print('face_location: ' + str(imdb[5][0][i][0]))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img
