from abc import ABC, abstractmethod


class EDAInterface(ABC):
    @abstractmethod
    def run(self, data):
        pass
