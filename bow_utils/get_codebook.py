import sys
import numpy as np

import pickle
from sklearn.cluster import KMeans

from bow_utils.get_image_descriptors import get_image_descriptors
from config import USE_PRECOMPUTED_CODEBOOK, CODEBOOK_PATH, \
    FEATURES_VECTOR_LENGTH, CLUSTERS_COUNT, KMEANS_ITERATIONS


def get_codebook(images_sets, sift, load=USE_PRECOMPUTED_CODEBOOK):
    if load:
        with open(CODEBOOK_PATH, 'rb') as file:
            codebook = pickle.load(file)
    else:
        codebook = _compute_codebook(images_sets, sift)
        with open(CODEBOOK_PATH, 'wb') as file:
            pickle.dump(codebook, file, pickle.HIGHEST_PROTOCOL)

    return codebook


def _compute_codebook(images_sets, sift):
    print('COMPUTING CODEBOOK:')

    print('  > PROCESSING CATEGORIES:')

    descriptors = np.empty(shape=(0, FEATURES_VECTOR_LENGTH))

    for set_name, set_path in images_sets.items():
        print(f'    > {set_name}')
        category_descriptors = _get_set_descriptors(set_path, sift)
        descriptors = np.vstack((descriptors, category_descriptors))

    print('    DONE')
    print('')

    print('  > BUILDING CODEBOOK:')

    codebook = KMeans(n_clusters=CLUSTERS_COUNT, n_init=KMEANS_ITERATIONS)
    codebook.fit(descriptors)

    print('    DONE')
    print('')

    return codebook


def _get_set_descriptors(set_path, sift):
    all_descriptors = np.empty(shape=(0, FEATURES_VECTOR_LENGTH))

    for image_path in set_path.iterdir():
        str = f'        processing image: {image_path.name}'
        print('\r', end='')
        print(str, end='')
        sys.stdout.flush()

        descriptors = get_image_descriptors(image_path, sift)

        if descriptors is not None:
            all_descriptors = np.vstack((all_descriptors, descriptors))

    print('\r', end='')
    print('      DONE')

    return all_descriptors
