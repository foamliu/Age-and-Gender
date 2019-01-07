import torch

image_h = image_w = image_size = 224
channel = 3
epochs = 10000
patience = 10

# Model parameters
num_classes = 101
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
lr = 1e-4  # learning rate for encoder if fine-tuning
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
DATA_DIR = 'data'
IMG_DIR = 'data/imdb_crop'
pickle_file = DATA_DIR + '/' + 'imdb-gender-age101.pkl'
