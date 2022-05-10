# %%
from copy import copy
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

# %%
def is_valid_pos(pos, board_length=55, board_width=40):
    return (0 <= pos[0] <= board_length - 1) and (0 <= pos[1] <= board_width - 1)


def search_path_keys(pos_stt:tuple, pos_dst:tuple, poss_wall:set, poss_danger_rate=dict(), epsilon=1e-6):
    # Please ensure max(poss_subscore) * epsilon < 1
    # A-star algorithm
    if (pos_stt == pos_dst):
        return ''
    searched = set()
    pos_infos = {pos_stt: (0, ham_dist(pos_stt, pos_dst), '')}
    while (len(pos_infos) > 0):
        pos_cur = min(pos_infos, key=lambda k:pos_infos[k][1])
        if (pos_cur == pos_dst):
            return pos_infos[pos_cur][2]
        searched.add(pos_cur)
        info_cur = pos_infos.pop(pos_cur)
        for dir in DIRECTIONS:
            pos_tmp = add_c(pos_cur, dir)
            if (pos_tmp not in searched and pos_tmp not in poss_wall and is_valid_pos(pos_tmp)):
                if (pos_tmp in poss_danger_rate):
                    subscore = poss_danger_rate[pos_tmp] * epsilon
                else:
                    subscore = 0.0
                pos_infos[pos_tmp] = (info_cur[0]+1, info_cur[0]+1+ham_dist(pos_cur, pos_dst)+subscore, info_cur[2]+dir2key(dir))
    return None
