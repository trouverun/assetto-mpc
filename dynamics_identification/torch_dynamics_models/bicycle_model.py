import torch.nn
from abc import ABC, abstractmethod


class BicycleModel(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def setup_params(self, x):
        pass

    @abstractmethod
    def extract_params(self):

        pass
    @abstractmethod
    def get_variable_scalers(self):

        pass
    @abstractmethod
    def get_output_weights(self):

        pass
    @abstractmethod
    def get_constraint_costs(self):
        pass

    @abstractmethod
    def extract_inputs_outputs(self, data, source_dirs):
        pass