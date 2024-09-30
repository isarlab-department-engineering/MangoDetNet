from torch.utils.data import Dataset
from abc import abstractmethod
import os


class TrainingDataset(Dataset):

    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    def __init__(self, config, mode, name):
        self.config = config
        self.name = name
        self.mode = mode
        self.train_data = None
        self.test_data = None
        self.valid_data = None
        self.read_data()

    @abstractmethod
    def read_data(self):
        raise ("Not implemented - this is an abstract method")

    @abstractmethod
    def print_info(self):
        raise ("Not implemented - this is an abstract method")

    def __len__(self):
        if self.mode == TrainingDataset.TRAIN:
            return len(self.train_data)
        elif self.mode == TrainingDataset.VALIDATION:
            return len(self.valid_data)
        elif self.mode == TrainingDataset.TEST:
            return len(self.test_data)
        else:
            raise ("Unknown MODE")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.mode == TrainingDataset.TRAIN:
            img, target, name = self.train_data[index].read_features(), self.train_data[index].read_labels(), os.path.basename(self.train_data[index].read_name())
        elif self.mode == TrainingDataset.VALIDATION:
            img, target, name = self.valid_data[index].read_features(), self.valid_data[index].read_labels(), os.path.basename(self.valid_data[index].read_name())
        elif self.mode == TrainingDataset.TEST:
            img, target, name = self.test_data[index].read_features(), self.test_data[index].read_labels(), os.path.basename(self.test_data[index].read_name())
        else:
            raise ("Unknown MODE")

        return img, target, name





