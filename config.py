import torch

image_w = 96
image_h = 112
channel = 3
epochs = 10000
patience = 10

# Model parameters
dropout = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
train_split = 0.95
age_num_classes = 101
gen_num_classes = 2

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 40
workers = 1  # for data-loading; right now, only 1 works with h5py
lr = 1e-4  # learning rate
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
DATA_DIR = 'data'
IMG_DIR = 'data/imdb_crop'
pickle_file = DATA_DIR + '/' + 'imdb-gender-age101.pkl'
