from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    @abstractmethod
    def plot(self):
        pass