import h5py
import json
import os
import numpy as np
# import torch
from collections import OrderedDict
from copy import deepcopy
import time

class ReplayWrapper:
    def __init__(self, file_dir):
        self.idx = 0
        self.file = h5py.File(file_dir)
        self.max_idx = len(self.file['action']['cartesian_velocity'])

    def forward(self, observation):
        if self.idx < self.max_idx:
            cart = self.file['action']['cartesian_velocity'][self.idx]
            grp = self.file['action']['gripper_position'][self.idx]
            action = np.concatenate([cart, [grp]])
            print(action)
            self.idx += 1
        else:
            return np.zeros(7,)
        return action