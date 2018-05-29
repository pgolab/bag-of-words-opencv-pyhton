import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

from bow_utils.get_image_descriptors import get_image_descriptors
from config import CLUSTERS_COUNT, USE_PRECOMPUTED_HISTOGRAMS, HISTOGRAMS_PATH


def get_images_sets_histograms(images_sets, sift, codebook, load=USE_PRECOMPUTED_HISTOGRAMS, save=True):
    if load:
        with open(HISTOGRAMS_PATH, 'rb') as file:
            sets_histograms = pickle.load(file)
    else:
        sets_histograms = _compute_images_sets_histograms(images_sets, sift, codebook)

        if save:
            with open(HISTOGRAMS_PATH, 'wb') as file:
                pickle.dump(sets_histograms, file, pickle.HIGHEST_PROTOCOL)

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
            codes = codebook.predict(descriptors)
            histogram, _ = np.histogram(codes, CLUSTERS_COUNT)
            histograms = np.vstack((histograms, histogram))

    print('\r', end='')
    print('    DONE')

    return histograms


def _plot_mean_histograms(sets_histograms):
    categories_mean_histograms = {}
    max_histograms_value = 0

    for category, histograms in sets_histograms.items():
        categories_mean_histograms[category] = np.mean(histograms, axis=0)
        max_histograms_value = max(max_histograms_value, np.max(categories_mean_histograms[category]))

    plt.figure(1)

    for index, (category, histogram) in enumerate(categories_mean_histograms.items()):
        plt.subplot(len(categories_mean_histograms.keys()), 1, index + 1)
        plt.bar(range(CLUSTERS_COUNT), histogram, width=1)
        plt.xlim(0, CLUSTERS_COUNT - 1)
        plt.ylim(0, max_histograms_value)
        plt.title(category)

    plt.show()

