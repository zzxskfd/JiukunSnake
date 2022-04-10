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
RIGHT = (0, 1)
LEFT = (0, -1)
UP = (-1, 0)
DOWN = (1, 0)

DIRECTIONS = [RIGHT, LEFT, UP, DOWN]
DIRECTIONS_KEY =  ['d', 'a', 'w', 's']

def key2dir(key: str):
    if (key == 'w'):
        return UP
    if (key == 'a'):
        return LEFT
    if (key == 's'):
        return DOWN
    if (key == 'd'):
        return RIGHT
    raise ValueError(f'Error: invalid key: {key}')

def add_c(x: tuple, y: tuple):
    return (x[0]+y[0], x[1]+y[1])

