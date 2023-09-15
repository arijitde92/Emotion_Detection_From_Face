import torch
from torch.utils.data import Dataset
import cv2


class FERDataset(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        'Initialization'
        self.transform = transforms
        self.paths = data_paths
        self.labels = data_labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]

        # Load data and get label
        image = cv2.imread(self.paths[index])
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image.float(), torch.tensor(label)
