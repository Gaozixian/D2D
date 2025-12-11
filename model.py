import torch
from pyparsing import originalTextFor
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import pandas as pd
import yaml
import csv
import matplotlib.pyplot as plt
from LSTM.data_def import add_data_dimension
def pre_data(seq_len, path):
    pre_dimension = add_data_dimension(seq_len, path)
    pre_dimension.add_data_dimension()


class model_dataset(Dataset):
    def __init__(self, files_path):
        super().__init__()
        self.name = 'CTL'
        self.path = files_path
        self.data = pd.read_csv(self.path)[['time_frame',
                                            'speed_ago', 'speed_now',
                                            'yaw_ago', 'yaw_now',
                                            'accel_ago', 'accel_now',
                                            'brake_ago', 'brake_now']]
