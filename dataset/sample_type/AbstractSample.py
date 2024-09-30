from abc import ABCMeta, abstractmethod


class AbstractSample(metaclass=ABCMeta):

    def __init__(self, transforms=None):
        self._transforms = transforms
        
    @property
    def transforms(self):
        return self._transforms

    @abstractmethod
    def read_features(self):
        raise ("Not implemented - this is an abstract method")

    @abstractmethod
    def read_labels(self):
        raise ("Not implemented - this is an abstract method")