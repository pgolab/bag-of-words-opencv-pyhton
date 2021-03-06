from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data'

SOURCES_DATA_PATH = DATA_PATH / 'sources'
TRAIN_DATA_PATH = DATA_PATH / 'train'
TEST_DATA_PATH = DATA_PATH / 'test'

COMPUTED_DATA_PATH = DATA_PATH / 'computed'
CODEBOOK_PATH = COMPUTED_DATA_PATH / 'codebook.pkl'
HISTOGRAMS_PATH = COMPUTED_DATA_PATH / 'histograms.pkl'
CLASSIFIER_PATH = COMPUTED_DATA_PATH / 'classifier.pkl'

USE_PRECOMPUTED_CODEBOOK = False
USE_PRECOMPUTED_HISTOGRAMS = False

DRAW_DETECTED_KEYPOINTS = False

FEATURES_VECTOR_LENGTH = 128

TEST_SIZE = 2
TRAIN_SIZE = 10

CLUSTERS_COUNT = 50
KMEANS_ITERATIONS = 10
CATEGORIES = ['helicopter', 'grand_piano', 'brain']
