import cv2
import numpy as np
from sklearn.svm import SVC

import pickle

from bow_utils.get_images_sets_histograms import get_images_sets_histograms
from bow_utils.get_codebook import get_codebook
from config import CATEGORIES, \
    TRAIN_DATA_PATH, CLASSIFIER_PATH, \
    CLUSTERS_COUNT


def train_classifiers(categories):
    sift = cv2.xfeatures2d.SIFT_create()
    images_sets = _get_training_images_sets(categories)

    codebook = get_codebook(images_sets, sift)

    sets_histograms = get_images_sets_histograms(images_sets, sift, codebook)

    categories_classifier = _compute_svm_classifier(categories, sets_histograms)

    with open(CLASSIFIER_PATH, 'wb') as file:
        pickle.dump(categories_classifier, file, pickle.HIGHEST_PROTOCOL)


def _get_training_images_sets(categories):
    images_sets = {}

    for category in categories:
        images_sets[category] = TRAIN_DATA_PATH / category

    return images_sets


def _compute_svm_classifier(categories, sets_histograms):
    print('TRAINING CLASSIFIER:')

    # ToDo: prepare test histograms for classification
    # combine all histograms and build labels array
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html

    # ToDo: train classifier
    # http://scikit-learn.org/stable/modules/svm.html

    print('DONE')

    return None


if __name__ == "__main__":
    train_classifiers(CATEGORIES)
