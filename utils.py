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

def L_inf_dist(x: tuple, y: tuple):
    return max(abs(x[0] - y[0]), abs(x[1] - y[1]))

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
    # info: (known dist, known_estimated dist, keys)
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


def search_local_cover_path(pos_center:tuple, dks_to_cover:list, pos_stt:tuple, poss_wall:set, poss_searched:set):
    # print(f'pos_center = {pos_center}')
    # print(f'dks_to_cover = {dks_to_cover}')
    # print(f'pos_stt = {pos_stt}')
    # print(f'poss_searched = {poss_searched}')
    assert(pos_stt in poss_searched)
    assert(L_inf_dist(pos_center, pos_stt) <= 1)
    assert(len(dks_to_cover) > 0)
    poss_to_cover = [add_c(pos_center, key2dir(dk)) for dk in dks_to_cover]
    def is_valid_tmp(pos):
        return is_valid_pos(pos) and L_inf_dist(pos_center, pos) <= 1
    best_act = None
    best_score = 0
    poss_cur = poss_searched
    def iter_acts(act_cur, pos_cur, score_sum, best_score, best_act):
        # Calculate current score (number of positions covered)
        score_sum_cur = score_sum + (1 if pos_cur in poss_to_cover else 0)
        if (score_sum_cur > best_score or (score_sum_cur == best_score and len(act_cur) < len(best_act))):
            best_score = score_sum_cur
            best_act = act_cur
        # Iterate all directions
        for dk in DIRECTIONS_KEY:
            next_pos = add_c(pos_cur, key2dir(dk))
            if (is_valid_tmp(next_pos) and next_pos not in poss_cur and next_pos not in poss_wall):
                poss_cur.add(next_pos)
                best_score, best_act = iter_acts(act_cur+dk, next_pos, score_sum_cur, best_score, best_act)
                poss_cur.remove(next_pos)
        return best_score, best_act

    best_score, best_act = iter_acts('', pos_stt, 0, best_score, best_act)
    # print(f'best_act = {best_act}')
    return best_act


# [20220521]
def search_path_keys_multi_target(pos_stt:tuple, poss_dst:set, poss_wall:set, max_dist:int):
    # BFS
    searched = set()
    to_search = {pos_stt: ''}   # pos -> keys
    step = 0
    while (len(to_search) > 0 and step <= max_dist):
        to_search_next = dict()
        for pos, keys in to_search.items():
            if (pos in poss_dst):
                return keys
            searched.add(pos)
            for dir in DIRECTIONS:
                pos_tmp = add_c(pos, dir)
                if (pos_tmp not in searched and pos_tmp not in poss_wall):
                    to_search_next[pos_tmp] = keys + dir2key(dir)
        # del to_search
        to_search = to_search_next
        step += 1
    return None

