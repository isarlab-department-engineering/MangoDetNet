from dataset.Dataset import TrainingDataset
from dataset.sample_type.FruitSample import FruitSample
import csv
import os
import os.path

import numpy as np
import torch


def read_object_labels_csv(file, header=True):
    items = []
    num_categories = 0
    # print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                items.append(item)
            rownum += 1
    return items

# ---- Fruit Presence/Absence Dataset
class FruitPADataset(TrainingDataset):

    def __init__(self, config, mode, name, data_root, transforms=None):
        self.opt = config
        self.transforms = transforms
        self.root = data_root
        self.path_devkit = os.path.join(data_root, 'Devkit')
        self.path_images = os.path.join(data_root, 'Devkit', 'JPEGImages')
        super(FruitPADataset, self).__init__(config, mode, name)

    def load_set(self, set):
        path_csv = os.path.join(self.path_devkit, 'BinaryAnnotations')
        file_csv = os.path.join(path_csv, set + '.csv')
        print(file_csv)

        items_names = read_object_labels_csv(file_csv)
        print('[dataset] Fruit Presence Absence set=%s number of images=%d' % (set, len(items_names)))

        items = []

        for item in items_names:

            path, target_count = item

            target_count = target_count.sum().unsqueeze(dim=0)
            if (torch.autograd.Variable(target_count)).data[0] > 0:
                target_class = torch.FloatTensor([1.0])
            else:
                target_class = torch.FloatTensor([0.0])

            img_path = os.path.join(self.path_images, path + '.jpg')
            items.append(FruitSample(img_path=img_path, label=target_class, transforms=self.transforms))

        return items

    def load_test_set(self, set):
        path_csv = os.path.join(self.path_devkit, 'CSV_Test_Annotations')
        items_names = os.listdir(path_csv)

        print('[dataset] Fruit Presence Absence set=%s number of images=%d' % (set, len(items_names)))

        items = []

        for item in items_names:
            f_path = os.path.join(path_csv, item)
            with open(f_path, 'r', encoding='cp1252') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip the headers
                labels = []
                for row in reader:
                    labels.append(row)
                # f.close()
                labels = np.asarray(labels).astype(np.float32)
                labels = torch.from_numpy(labels)
                img_path = os.path.join(self.path_images, item[:-4] + '.jpg')
                items.append(FruitSample(img_path=img_path, label=labels, transforms=self.transforms))

        return items


    def print_info(self):
        print("FruitDataset print_info: Not yet implemented")

    def read_data(self):
        # ----- load training_set
        if self.mode == TrainingDataset.TRAIN:
            self.train_data = self.load_set('train')
        # ----- load validation_set
        if self.mode == TrainingDataset.VALIDATION:
            self.valid_data = self.load_set('val')
            # ----- load test_set
        # if self.mode == TrainingDataset.TEST:
        #     self.test_data = self.load_set('test')


class FruitCountDataset(TrainingDataset):

    def __init__(self, config, mode, name, data_root, transforms=None):
        self.transforms = transforms
        self.root = data_root
        self.path_devkit = os.path.join(data_root, 'Devkit')
        self.path_images = os.path.join(data_root, 'Devkit', 'JPEGImages')
        super(FruitCountDataset, self).__init__(config, mode, name)

    def load_set(self, set):
        path_csv = os.path.join(self.path_devkit, 'NumericAnnotations')
        file_csv = os.path.join(path_csv, set + '.csv')
        print(file_csv)

        items_names = read_object_labels_csv(file_csv)
        print('[dataset] Fruit Count set=%s number of images=%d' % (set, len(items_names)))

        items = []

        for item in items_names:

            path, target_count = item

            target_count = torch.FloatTensor(target_count)

            img_path = os.path.join(self.path_images, path + '.jpg')
            items.append(FruitSample(img_path=img_path, label=target_count, transforms=self.transforms))

        return items

    def print_info(self):
        print("FruitDataset print_info: Not yet implemented")

    def read_data(self):
        # ----- load training_set
        if self.mode == TrainingDataset.TRAIN:
            self.train_data = self.load_set('train')
        # ----- load validation_set
        if self.mode == TrainingDataset.VALIDATION:
            self.valid_data = self.load_set('val')
            # ----- load test_set
        if self.mode == TrainingDataset.TEST:
            self.test_data = self.load_set('test')


class FruitBboxDataset(TrainingDataset):

    def __init__(self, config, mode, name, data_root, transforms=None):
        self.transforms = transforms
        self.root = data_root
        self.path_devkit = os.path.join(data_root, 'Devkit')
        self.path_images = os.path.join(data_root, 'Devkit', 'JPEGImages')
        super(FruitBboxDataset, self).__init__(config, mode, name)


    def load_set(self, set):
        path_csv = os.path.join(self.path_devkit, 'BBoxAnnotations', set)
        items_names = os.listdir(path_csv)

        print('[dataset] Fruit BBox set=%s number of images=%d' % (set, len(items_names)))

        items = []

        for item in items_names:
            f_path = os.path.join(path_csv, item)
            with open(f_path, 'r', encoding='cp1252') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip the headers
                labels = []
                for row in reader:
                    labels.append(row)
                # f.close()
                labels = np.asarray(labels).astype(np.float32)
                labels = torch.from_numpy(labels)
                img_path = os.path.join(self.path_images, item[:-4] + '.jpg')
                items.append(FruitSample(img_path=img_path, label=labels, transforms=self.transforms))

        return items

    def print_info(self):
        print("FruitDataset print_info: Not yet implemented")

    def read_data(self):
        # ----- load training_set
        if self.mode == TrainingDataset.TRAIN:
            self.train_data = self.load_set('train')
        # ----- load validation_set
        if self.mode == TrainingDataset.VALIDATION:
            self.valid_data = self.load_set('val')
            # ----- load test_set
        if self.mode == TrainingDataset.TEST:
            self.test_data = self.load_set('test')
