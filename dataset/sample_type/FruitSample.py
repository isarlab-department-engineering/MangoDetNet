import cv2
import torch

from dataset.sample_type.AbstractSample import AbstractSample

# Fruit sample
class FruitSample(AbstractSample):

    def __init__(self, img_path, label, is_grayscale=False, transforms=None):
        super(FruitSample, self).__init__(transforms)
        self.img_path = img_path
        self.is_grayscale = is_grayscale
        self.label = label
        self.features = self.read_img()

    def read_img(self):
        if self.is_grayscale:
            img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        return img

    def read_features(self):
        return self.features

    def read_labels(self):
        return self.label

    def read_name(self):
        return self.img_path
