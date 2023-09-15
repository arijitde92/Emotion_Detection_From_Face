import os
from typing import List
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm

random.seed(42)
DATA_ROOT = 'data'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
emotions = os.listdir(TRAIN_DIR)
label_dict = dict()
for idx, emotion in enumerate(emotions):
    label_dict[emotion] = idx
print(label_dict)


def fill_labels(image_paths: List):
    labels = []
    for img_path in tqdm(image_paths):
        label = img_path.split(os.sep)[2]
        labels.append(label_dict.get(label))
    return labels


def get_paths_and_labels(val_split=0.2, shuffle=True):
    train_image_paths = glob(os.path.join(TRAIN_DIR, '**', '*.jpg'))
    test_image_paths = glob(os.path.join(TRAIN_DIR, '**', '*.jpg'))
    n_train_images = len(train_image_paths)
    if shuffle:
        random.shuffle(train_image_paths)
        random.shuffle(test_image_paths)
    else:
        train_image_paths.sort()
        test_image_paths.sort()
    print(f"Found {len(train_image_paths)} train images\nSplitting into validation and training with val={val_split}")
    val_image_paths = train_image_paths[:int(val_split*n_train_images)]
    training_image_paths = train_image_paths[int(val_split*n_train_images):]
    train_labels = fill_labels(training_image_paths)
    val_labels = fill_labels(val_image_paths)
    test_labels = fill_labels(test_image_paths)
    print("Total Training images:", len(training_image_paths))
    print("Total Validation images:", len(val_image_paths))
    print("Total Test images:", len(test_image_paths))
    return (training_image_paths, train_labels), (val_image_paths, val_labels), (test_image_paths, test_labels)


def get_transforms():
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.8),
            A.ShiftScaleRotate(shift_limit=(-0.08, 0.08), scale_limit=(-0.001, 0.001), rotate_limit=(-15, 15),
                               interpolation=1, border_mode=2, p=0.9),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose([ToTensorV2()])
    return train_transform, test_transform


def get_num_classes():
    return len(emotions)
