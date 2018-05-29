import cv2

import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

from bow_utils.get_codebook import get_codebook
from bow_utils.get_images_sets_histograms import get_images_sets_histograms
from config import CATEGORIES, \
    TEST_DATA_PATH, CLASSIFIER_PATH


def test_classifiers(categories):
    with open(CLASSIFIER_PATH, 'rb') as file:
        classifier = pickle.load(file)

    sift = cv2.xfeatures2d.SIFT_create()
    images_sets = _get_test_images_sets(categories)
    codebook = get_codebook(categories, sift, load=True)
    sets_histograms = get_images_sets_histograms(images_sets, sift, codebook, load=False, save=False)

    set_labels = []
    predicted_labels = []

    print(f'CLASSIFYING')

    for category in categories:
        print(f'  > {category}:')

        predictions = classifier.predict(sets_histograms[category])
        set_labels += [category] * len(sets_histograms[category])
        predicted_labels += predictions.tolist()

        print(f'  DONE')

    print(set_labels)
    print(predicted_labels)

    _draw_confusion_matrix(set_labels, predicted_labels, images_sets.keys())


def _get_test_images_sets(categories):
    images_sets = {}

    for category in categories:
        images_sets[category] = TEST_DATA_PATH / category

    return images_sets


def _draw_confusion_matrix(set_labels, predicted_labels, classes):
    matrix = confusion_matrix(set_labels, predicted_labels)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confuction matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, f'{matrix[i,j]:.2f}',
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")

    plt.show()


if __name__ == "__main__":
    test_classifiers(CATEGORIES)
