from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def initialize(self, track, reference):
        pass

    @abstractmethod
    def get_control(self, initial_state, max_speed=None):
        pass

    @abstractmethod
    def learn_from_data(self, inputs, outputs):
        pass

    @abstractmethod
    def kill(self):
        pass
