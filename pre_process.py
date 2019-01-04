import tarfile
import scipy.io


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    extract('data/imdb_meta.tar')
    extract('data/imdb_crop.tar')

    mat = scipy.io.loadmat('data/imdb/imdb.mat')
