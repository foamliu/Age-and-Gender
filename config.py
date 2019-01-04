import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = 'data/imdb_crop'
feature_size = 103
