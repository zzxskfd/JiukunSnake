# %%
import os
import numpy as np
import json
from struct import Struct
# from dataclasses import dataclass


def load_json(path):
    with open(path, 'r') as file:
        res = json.load(file)
    return res

def save_json(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file, indent=4)

def create_folder(dir):
    if (not os.path.exists(dir)):
        os.makedirs(dir)

# %%
