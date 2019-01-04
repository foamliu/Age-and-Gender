import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename)
    tar.extractall()
    tar.close()


if __name__ == "__main__":
    extract('data/imdb_meta.tar')
    extract('data/imdb_crop.tar')
