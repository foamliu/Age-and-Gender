import pickle

import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt
from config import *

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    age = data['age']
    gender = data['gender']
    sns.distplot(age, kde=True, rug=True)
    # sns.distplot(gender, kde=True, rug=True)
    plt.show()
    print("Age size: " + str(len(age)))
