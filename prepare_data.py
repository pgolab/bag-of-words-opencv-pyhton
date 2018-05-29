from config import \
    SOURCES_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, \
    CATEGORIES, TEST_SIZE, TRAIN_SIZE
import shutil
from sklearn.model_selection import train_test_split


def generate_training_and_test_data(categories):
    categories_lists = _get_categories_lists(categories)

    _prepare_path(TRAIN_DATA_PATH)
    _prepare_path(TEST_DATA_PATH)

    for category, images_list in categories_lists.items():
        # ToDo: split images into test and training set
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        _copy_files_to([], TRAIN_DATA_PATH / category)
        _copy_files_to([], TEST_DATA_PATH / category)


def _get_categories_lists(categories):
    categories_lists = {}

    for category in categories:
        category_dir = SOURCES_DATA_PATH / category
        category_list = list(filter(lambda path: not path.name.startswith('.'), category_dir.iterdir()))
        categories_lists[category] = category_list

    return categories_lists


def _prepare_path(path):
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        for sub_path in path.iterdir():
            if sub_path.name.startswith('.'):
                continue

            shutil.rmtree(sub_path)


def _copy_files_to(files_list, destination):
    destination.mkdir()
    for file in files_list:
        shutil.copy(file, destination)


if __name__ == "__main__":
    generate_training_and_test_data(CATEGORIES)
