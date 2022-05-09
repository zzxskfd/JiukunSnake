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

# def os_sorted(filelist):
#     max_len = max([len(f) for f in filelist])
#     for i in range(len(filelist)):
#         name = '.'.join(filelist[i].split('.')[:-1])
#         sfx = filelist[i].split('.')[-1]
#         filelist[i] = name + ''.join(['~']*(max_len - len(name))) + '.' + sfx
#     filelist = sorted(filelist)
#     for i in range(len(filelist)):
#         filelist[i] = filelist[i].replace('~', '')
#     return filelist

# %%
RIGHT = (1, 0)
LEFT = (-1, 0)
UP = (0, 1)
DOWN = (0, -1)

DIRECTIONS = [RIGHT, LEFT, UP, DOWN]
DIRECTIONS_KEY =  ['d', 'a', 'w', 's']

KEY2DIR_DICT = {
    'w': UP,
    'a': LEFT,
    's': DOWN,
    'd': RIGHT,
}
def key2dir(key: str):
    return KEY2DIR_DICT[key]

DIR2KEY_DICT = {
    UP: 'w',
    LEFT: 'a',
    DOWN: 's',
    RIGHT: 'd',
}
def dir2key(dir: tuple):
    return DIR2KEY_DICT[dir]

REVERSE_KEY_DICT = {
    'w': 's',
    's': 'w',
    'a': 'd',
    'd': 'a',
}
def r_key(key: str):
    return REVERSE_KEY_DICT[key]

def add_c(x: tuple, y: tuple):
    return (x[0]+y[0], x[1]+y[1])

def minus_c(x: tuple, y: tuple):
    return (x[0]-y[0], x[1]-y[1])

def ham_dist(x: tuple, y: tuple):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

def get_reverse_path_keys(poss: list):
    assert(len(poss) > 1)
    res = ''
    for i in range(len(poss) - 1):
        res += dir2key(minus_c(poss[i+1], poss[i]))
    return res
