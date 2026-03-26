from abc import ABC, abstractmethod

from antifraud.utils.logger import get_logger


class TaskBase(ABC):
    def __init__(self, loader):
        self.loader = loader
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self):
        pass
