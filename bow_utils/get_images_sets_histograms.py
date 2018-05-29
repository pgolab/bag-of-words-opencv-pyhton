import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

from bow_utils.get_image_descriptors import get_image_descriptors
from config import CLUSTERS_COUNT, USE_PRECOMPUTED_HISTOGRAMS, HISTOGRAMS_PATH


def get_images_sets_histograms(images_sets, sift, codebook, load=USE_PRECOMPUTED_HISTOGRAMS, save=True):
    if load:
        # ToDo: load saved sets histograms
        # https://docs.python.org/3.6/library/pickle.html
        sets_histograms = None
    else:
        sets_histograms = _compute_images_sets_histograms(images_sets, sift, codebook)
        # ToDo: save sets histograms
        # https://docs.python.org/3.6/library/pickle.html

    _plot_mean_histograms(sets_histograms)

    return sets_histograms


def _compute_images_sets_histograms(images_sets, sift, codebook):
    print('COMPUTING SETS HISTOGRAMS:')

    sets_histograms = {}

    for set_name, set_path in images_sets.items():
        print(f'  > {set_name}')
        sets_histograms[set_name] = _get_images_set_histograms(set_path, sift, codebook)

    print('  DONE')
    print('')

    return sets_histograms


def _get_images_set_histograms(set_path, sift, codebook):
    histograms = np.empty((0, CLUSTERS_COUNT))

    for image_path in set_path.iterdir():
        str = f'      processing image: {image_path.name}'
        print('\r', end='')
        print(str, end='')
        sys.stdout.flush()

        descriptors = get_image_descriptors(image_path, sift)

        if descriptors is not None:
            # ToDo: build histogram using codebook
            # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            pass

    print('\r', end='')
    print('    DONE')

    return histograms


def _plot_mean_histograms(sets_histograms):
    categories_mean_histograms = {}
    max_histograms_value = 0

    for category, histograms in sets_histograms.items():
        # ToDo: calculate mean histogram
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        pass

    # ToDo: plot histogram
    # https://matplotlib.org/users/pyplot_tutorial.html

