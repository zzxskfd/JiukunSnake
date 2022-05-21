# %%
import numpy as np
import random
from scipy.stats import rankdata
from copy import copy

from utils import DIRECTIONS, DIRECTIONS_KEY, dir2key, get_reverse_path_keys, ham_dist, key2dir, add_c, minus_c, r_key, search_local_cover_path, search_path_keys

def transfer_poss(poss):
    return [tuple(pos) for pos in poss]
    # return [(int(pos[0]), int(pos[1])) for pos in poss]

def AI0(Num_, GameInfo_, params_=None, debug=False):
    # Default params
    params = {
        'score_hit_wall': -10000,
        'score_hit_self_strong': -15000,
        'score_hit_others_body': -20000,
        'score_sugar': 3.25,
        'score_sugar_alpha': 14.84,   # >= 1
        'score_sugar_beta': 1.61,    # >= 0
        'score_sugar_gamma': 0.5,   # >= 0
        'score_speed': 10.78,
        'score_speed_alpha': 18.2,   # >= 1
        'score_speed_beta': 0.3,    # >= 0
        'score_speed_gamma': 1.1,   # >= 0
        'score_strong': 6.01,
        'score_strong_alpha': 22.6,   # >= 1
        'score_strong_beta': 7.0,    # >= 0
        'score_strong_gamma': 0.53,   # >= 0
        'score_double': 6.4,
        'score_double_alpha': 8.69,   # >= 1
        'score_double_beta': 9.41,    # >= 0
        'score_double_gamma': 0.63,   # >= 0
        'score_empty_grid': 0,
        'score_center_alpha_x': -0.4,    # if < 0, center is encouraged
        'score_center_alpha_y': -0.42,    # if < 0, center is encouraged
        'score_enemy_stronger': -23,       # should be negative
        'score_enemy_weaker': 33.43,          # should be positive
        'score_enemy_both_strong': -13.06,    # should be negative to discourage snake to move too much
        'score_enemy_stronger_dist': -7.29,      # should be negative
        'score_enemy_weaker_dist': 30.09,         # should be positive
        'score_enemy_both_strong_dist': -1.33,   # should be negative to discourage snake to move too much
        'score_discount_rate': 0.75,     # should be in [0.0, 1.0]
        'score_constant': 73.4,
    }
    
    if (params_ is not None):
        params.update(params_)
    # If dead, return 'd'
    players = GameInfo_["gameinfo"]["Player"]
    game_map = GameInfo_["gameinfo"]["Map"]
    player_self = players[Num_]
    if(player_self["IsDead"]):
        return "d"
    # Basic attributes
    player_num = len(players)
    map_length = 55
    map_width = 40
    speed = player_self["Speed"]
    is_strong = player_self["Prop"]["strong"] > 0
    is_double = player_self["Prop"]["double"] > 0
    rank_len = rankdata([players[i]["Score_len"] for i in range(player_num)])[Num_]
    rank_kill = rankdata([players[i]["Score_kill"] for i in range(player_num)])[Num_]
    rank_time = rankdata([players[i]["Score_time"] for i in range(player_num)])[Num_]
    # Positions
    PositionHead = tuple(game_map["SnakePosition"][Num_][0])
    WallPositions = set(transfer_poss(game_map["WallPosition"]))
    SugarPositions = set(transfer_poss(game_map["SugarPosition"]))
    SpeedPositions = set(transfer_poss(game_map["PropPosition"][0]))
    StrongPositions = set(transfer_poss(game_map["PropPosition"][1]))
    DoublePositions = set(transfer_poss(game_map["PropPosition"][2]))
    SnakeHitPositions = []
    # Valid snake bodies
    alive = 6
    for i_snake in range(player_num):
        if(players[i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
            alive -= 1
        else:
            # The last element of snake will not hit player if SaveLength == 0
            if (players[i_snake]["SaveLength"] == 0):
                SnakeHitPositions.append(set(transfer_poss(game_map["SnakePosition"][i_snake][:-1])))
            else:
                SnakeHitPositions.append(set(transfer_poss(game_map["SnakePosition"][i_snake])))
    # Walls and snake bodies
    wall_and_snake_hit_poss = WallPositions.union(*SnakeHitPositions)
    # Strong or weak enemies
    next_max_lens = []
    for i_snake in range(player_num):
        next_max_lens.append(len(game_map["SnakePosition"][i_snake]) + min(players[i_snake]["Speed"], players[i_snake]["SaveLength"]))
    power_levels = []
    for i_snake in range(player_num):
        if (i_snake == Num_):
            power_levels.append('self')
            continue
        if (players[i_snake]["IsDead"]):
            power_levels.append('dead')
            continue
        if (not is_strong):
            if (players[i_snake]["Prop"]["strong"] > 0):
                power_levels.append('stronger')
            elif (next_max_lens[i_snake] > next_max_lens[Num_]):
                power_levels.append('stronger')
            elif (next_max_lens[i_snake] == next_max_lens[Num_]):
                power_levels.append('equal')
            else:
                power_levels.append('weaker')
        else:
            if (players[i_snake]["Prop"]["strong"] > 0):
                power_levels.append('both_strong')
            else:
                power_levels.append('weaker')
    # Score map considering enemies
    score_map_enemy = dict()
    for i_snake in range(player_num):
        if (power_levels[i_snake] == 'stronger'):
            level_tmp = params['score_enemy_stronger']
        elif (power_levels[i_snake] == 'both_strong'):
            level_tmp = params['score_enemy_both_strong']
        elif (power_levels[i_snake] == 'weaker'):
            level_tmp = params['score_enemy_weaker']
        else:
            continue

        poss_cur = {tuple(game_map["SnakePosition"][i_snake][0])}
        searched = copy(wall_and_snake_hit_poss)
        for step in range(players[i_snake]["Speed"]):
            if (len(poss_cur) == 0):
                break
            searched = searched | poss_cur
            poss_nxt = set()
            for pos in poss_cur:
                # Calc score here
                if (step > 0):
                    if (pos in score_map_enemy):
                        score_map_enemy[pos] += level_tmp / step
                    else:
                        score_map_enemy[pos] = level_tmp / step
                for dir in DIRECTIONS:
                    pos_tmp = add_c(pos, dir)
                    if (pos_tmp not in searched):
                        poss_nxt.add(pos_tmp)
            poss_cur = poss_nxt

    # Basic scores that doesnt depend on moving path
    pos_score_cache = dict()
    def get_pos_score(pos):
        if (pos in pos_score_cache):
            return pos_score_cache[pos]
        if (pos in WallPositions):
            if alive == 1 and rank_len == 6:
                return -params['score_hit_wall']
            # print('Hit wall:', pos)
            return params['score_hit_wall']
        res = 0
        for i_snake in range(len(players)):
            if (pos in SnakeHitPositions[i_snake]):
                if (i_snake == Num_ and is_strong):
                    res += params['score_hit_self_strong']
                elif (i_snake == Num_):
                    res += params['score_hit_wall']
                else:
                    res += params['score_hit_others_body']

        if (res < 0):
            return res
        if (pos in SugarPositions):
            alpha = params['score_sugar_alpha']
            beta = params['score_sugar_beta']
            gamma = params['score_sugar_gamma']
            score_tmp = params['score_sugar'] * max(1, alpha - beta * player_self["Score_len"] - gamma * rank_len)
            if (is_double):
                score_tmp *= 2
            res += score_tmp
        elif (pos in SpeedPositions):
            alpha = params['score_speed_alpha']
            beta = params['score_speed_beta']
            gamma = params['score_speed_gamma']
            res += params['score_speed'] * max(1, alpha - beta * speed - gamma * player_self["Prop"]["speed"])
        elif (pos in StrongPositions):
            alpha = params['score_strong_alpha']
            beta = params['score_strong_beta']
            gamma = params['score_strong_gamma']
            res += params['score_strong'] * max(1, alpha - beta * is_strong - gamma * player_self["Prop"]["strong"])
        elif (pos in DoublePositions):
            alpha = params['score_double_alpha']
            beta = params['score_double_beta']
            gamma = params['score_double_gamma']
            res += params['score_double'] * max(1, alpha - beta * is_double - gamma * player_self["Prop"]["double"])
        else:
            res += params['score_empty_grid']
        # Positions closer to center is better
        res += abs(pos[0] - (map_length-1)/2) * params['score_center_alpha_x']
        res += abs(pos[1] - (map_width-1)/2) * params['score_center_alpha_y']
        # score_map_enemy
        if (pos in score_map_enemy):
            res += score_map_enemy[pos]
        # According to distance of enemy
        for i_snake in range(player_num):
            if (power_levels[i_snake] == 'stronger'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_stronger_dist'] / dist_tmp
            elif (power_levels[i_snake] == 'weaker'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_weaker_dist'] / dist_tmp
            elif (power_levels[i_snake] == 'both_strong'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_both_strong_dist'] / dist_tmp

        res += params['score_constant']
        pos_score_cache[pos] = res        
        return res

    # max_search_step = 12
    max_search_step = 10
    hit_score_threshold = -10000
    act_values = dict()
    act_values_real = dict()        # For cases speed > length
    poss_cur = set([PositionHead])
    def iter_acts(act_cur, pos_cur, score_sum):
        if (len(act_cur) > 0):
            score_sum_cur = score_sum + get_pos_score(pos_cur)
            act_values[act_cur] = score_sum_cur
            # Consider cases speed > length, and add discount rate for future searches
            real_move_len = min(speed, len(act_cur))
            act_values_real[act_cur] = params['score_discount_rate'] * act_values[act_cur]\
                                       + (1-params['score_discount_rate']) * act_values[act_cur[:real_move_len]]
            if (real_move_len > next_max_lens[Num_]):
                act_values_real[act_cur] -= act_values[act_cur[:real_move_len - next_max_lens[Num_]]]
        else:
            score_sum_cur = 0
        if (len(act_cur) < max_search_step and score_sum_cur > hit_score_threshold):
            for dk in DIRECTIONS_KEY:
                next_pos = add_c(pos_cur, key2dir(dk))
                if (next_pos not in poss_cur):
                    poss_cur.add(next_pos)
                    iter_acts(act_cur+dk, next_pos, score_sum_cur)
                    poss_cur.remove(next_pos)
    
    iter_acts('', PositionHead, 0)
    # print('Num =', Num_)
    if (debug):
        print('act_values =', act_values)
        print('act_values_real =', act_values_real)
    score_best = max(act_values_real.values())
    ok_keys = [dk for dk in act_values_real if act_values_real[dk] == score_best]
    if (debug):
        print('ok_keys =', ok_keys)
    res_keys = random.choice(ok_keys)
    if (score_best > hit_score_threshold):
        # If there exists a path not hitting any snake body or wall
        return res_keys[:speed]

    # Calculate whether can escape and decide whether to escape
    self_length_total = len(game_map["SnakePosition"][Num_]) + players[Num_]["SaveLength"]
    self_strong_time = players[Num_]["Prop"]["strong"]
    self_speed_time = players[Num_]["Prop"]["speed"]
    # If there is a wall formed by body of self, hit self until have a way out.
    extra_tol_len = 0   # should be at least 1 if self length > 1
    best_dk = None
    for dk in DIRECTIONS_KEY:
        dir = key2dir(dk)
        pos_tmp = add_c(PositionHead, dir)
        if (pos_tmp in SnakeHitPositions[Num_]):
            body_ind = game_map["SnakePosition"][Num_].index((pos_tmp[0], pos_tmp[1]))
            if (body_ind > extra_tol_len):
                extra_tol_len = body_ind
                best_dk = dk
    if (debug):
        print('extra_tol_len =', extra_tol_len)
    if (extra_tol_len < 1):
        return res_keys
    # 2-round speed sum
    valid_speed_time = min(self_speed_time, 2)
    two_speed_sum = speed * valid_speed_time + (2 - valid_speed_time)
    # max_speed_sum: speed sum in (self_strong_time + 1) rounds
    valid_speed_time = min(self_speed_time, self_strong_time + 1)
    max_speed_sum = speed * valid_speed_time + (self_strong_time + 1 - valid_speed_time)
    if (debug):
        print('self_length_total - extra_tol_len =', self_length_total - extra_tol_len)
        print('speed =', speed)
        print('two_speed_sum =', two_speed_sum)
        print('max_speed_sum =', max_speed_sum)
    # Three steps escape
    def hit_self_keys(first_key, num_step):
        res = first_key
        while (len(res) < num_step):
            res += r_key(res[-1])
        return res

    if (speed >= self_length_total - extra_tol_len and players[Num_]["SaveLength"] == 0):
        # SaveLength must be 0! otherwise would be complicated
        return best_dk + get_reverse_path_keys(game_map["SnakePosition"][Num_][extra_tol_len:])
    elif (speed >= self_length_total - extra_tol_len):
        return hit_self_keys(best_dk, players[Num_]["SaveLength"])
    elif (two_speed_sum >= self_length_total - extra_tol_len):
        len_to_destroy = self_length_total - extra_tol_len - speed  # should be >= 1
        return hit_self_keys(best_dk, len_to_destroy)
    elif (max_speed_sum >= self_length_total - extra_tol_len):
        return hit_self_keys(best_dk, speed)
    else:
        # cannot escape
        return res_keys
    



# %% Greedy for current round
def AI_greedy_0(Num_, GameInfo_, params_=None, debug=False):
    # Default params
    params = {
        'score_hit_wall': -150000,
        'score_hit_self_strong': -100000,
        'score_hit_others_body': -200000,
        'score_sugar': 2.97,
        'score_sugar_alpha': 14.9,   # >= 1
        'score_sugar_beta': 1.73,    # >= 0
        'score_sugar_gamma': 0.49,   # >= 0
        'score_speed': 7.96,
        'score_speed_alpha': 18.22,   # >= 1
        'score_speed_beta': 0.28,    # >= 0
        'score_speed_gamma': 1.14,   # >= 0
        'score_strong': 6.32,
        'score_strong_alpha': 22.21,   # >= 1
        'score_strong_beta': 6.83,    # >= 0
        'score_strong_gamma': 0.9,   # >= 0
        'score_double': 7.66,
        'score_double_alpha': 10.32,   # >= 1
        'score_double_beta': 9.36,    # >= 0
        'score_double_gamma': 0.63,   # >= 0
        'score_empty_grid': 0,
        'score_center_alpha_x': -0.38,    # if < 0, center is encouraged
        'score_center_alpha_y': -0.41,    # if < 0, center is encouraged
        'score_enemy_stronger': -22.32,       # should be negative
        'score_enemy_weaker': 34.79,          # should be positive
        'score_enemy_both_strong': -12.95,    # should be negative to discourage snake to move too much
        'score_enemy_stronger_dist': -6.8,      # should be negative
        'score_enemy_weaker_dist': 29.64,         # should be positive
        'score_enemy_both_strong_dist': -2.05,   # should be negative to discourage snake to move too much
        'score_discount_rate': 0.74,     # should be in [0.0, 1.0]
        'score_constant': 73.95,
        'score_strong_enough': -7.74,
        'score_connect_0': -6.23,
        'score_connect_1': -6.25,
        'score_connect_2': 3.44,
        'score_connect_3': 5.46,
        'score_sugar_new': 2.97,
        'score_speed_new': 7.96,
        'score_strong_new': 6.32,
        'score_double_new': 7.66,
        'score_connect_2_new': 0,
        'w1': 0,
        'w2': 0,
        'b': 0,
    }
    
    if (params_ is not None):
        params.update(params_)
    # If dead, return 'd'
    players = GameInfo_["gameinfo"]["Player"]
    game_map = GameInfo_["gameinfo"]["Map"]
    player_self = players[Num_]
    if(player_self["IsDead"]):
        return "d"
    # Basic attributes
    player_num = len(players)
    map_length = 55
    map_width = 40
    speed = player_self["Speed"]
    is_strong = player_self["Prop"]["strong"] > 0
    is_double = player_self["Prop"]["double"] > 0
    rank_len = rankdata([players[i]["Score_len"] for i in range(player_num)])[Num_]
    rank_kill = rankdata([players[i]["Score_kill"] for i in range(player_num)])[Num_]
    rank_time = rankdata([players[i]["Score_time"] for i in range(player_num)])[Num_]
    # Positions
    PositionHead = tuple(game_map["SnakePosition"][Num_][0])
    WallPositions = set(transfer_poss(game_map["WallPosition"]))
    SugarPositions = set(transfer_poss(game_map["SugarPosition"]))
    SpeedPositions = set(transfer_poss(game_map["PropPosition"][0]))
    StrongPositions = set(transfer_poss(game_map["PropPosition"][1]))
    DoublePositions = set(transfer_poss(game_map["PropPosition"][2]))
    SnakeHitPositions = []
    # Valid snake bodies
    alive = 6
    for i_snake in range(player_num):
        if(players[i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
            alive -= 1
        else:
            # The last element of snake will not hit player if SaveLength == 0
            if (players[i_snake]["SaveLength"] == 0):
                SnakeHitPositions.append(set(transfer_poss(game_map["SnakePosition"][i_snake][:-1])))
            else:
                SnakeHitPositions.append(set(transfer_poss(game_map["SnakePosition"][i_snake])))
    # Walls and snake bodies
    wall_and_snake_hit_poss = WallPositions.union(*SnakeHitPositions)
    # Strong or weak enemies
    next_max_lens = []
    for i_snake in range(player_num):
        next_max_lens.append(len(game_map["SnakePosition"][i_snake]) + min(players[i_snake]["Speed"], players[i_snake]["SaveLength"]))
    power_levels = []
    for i_snake in range(player_num):
        if (i_snake == Num_):
            power_levels.append('self')
            continue
        if (players[i_snake]["IsDead"]):
            power_levels.append('dead')
            continue
        if (not is_strong):
            if (players[i_snake]["Prop"]["strong"] > 0):
                power_levels.append('stronger')
            elif (next_max_lens[i_snake] > next_max_lens[Num_]):
                power_levels.append('stronger')
            elif (next_max_lens[i_snake] == next_max_lens[Num_]):
                power_levels.append('equal')
            else:
                power_levels.append('weaker')
        else:
            if (players[i_snake]["Prop"]["strong"] > 0):
                power_levels.append('both_strong')
            else:
                power_levels.append('weaker')

    # Score map considering enemies
    score_map_enemy = dict()
    danger_map_for_kill = dict()
    for i_snake in range(player_num):
        if (power_levels[i_snake] == 'stronger'):
            level_tmp = params['score_enemy_stronger']
        elif (power_levels[i_snake] == 'both_strong'):
            level_tmp = params['score_enemy_both_strong']
        elif (power_levels[i_snake] == 'weaker'):
            level_tmp = params['score_enemy_weaker']
        else:
            continue

        poss_cur = {tuple(game_map["SnakePosition"][i_snake][0])}
        searched = copy(wall_and_snake_hit_poss)
        for step in range(players[i_snake]["Speed"]):
            if (len(poss_cur) == 0):
                break
            searched = searched | poss_cur
            poss_nxt = set()
            for pos in poss_cur:
                # Calc score here
                if (step > 0):
                    if (pos in score_map_enemy):
                        score_map_enemy[pos] += level_tmp / step
                    else:
                        score_map_enemy[pos] = level_tmp / step
                    # subscore_map_for_kill is used in manual killer process
                    if (players[i_snake]["Prop"]["strong"] > 0):
                        if (pos in danger_map_for_kill):
                            danger_map_for_kill[pos] += 1 / step
                        else:
                            danger_map_for_kill[pos] = 1 / step
                for dir in DIRECTIONS:
                    pos_tmp = add_c(pos, dir)
                    if (pos_tmp not in searched):
                        poss_nxt.add(pos_tmp)
            poss_cur = poss_nxt

    # [20220519] Danger map includes strong enemy body
    for i_snake in range(player_num):
        if (players[i_snake]["Prop"]["strong"] == 0):
            continue
        for pos in SnakeHitPositions[i_snake]:
            if (pos in danger_map_for_kill):
                danger_map_for_kill[pos] += 1
            else:
                danger_map_for_kill[pos] = 1

    # [20220510] Manual killer process
    def get_manual_kill_keys():
        if (not is_strong):
            return None
        kill_plans = []
        for i_snake in range(player_num):
            if (i_snake == Num_ or players[i_snake]["IsDead"] or players[i_snake]["Prop"]["strong"] > 0):
                continue
            # If target speed > length, ignore. (Leave it to normal path search)
            if (players[i_snake]["Speed"] > next_max_lens[i_snake]):
                continue
            # If too far, ignore
            target_head_pos = tuple(game_map["SnakePosition"][i_snake][0])
            if (ham_dist(PositionHead, target_head_pos) > speed + 1):
                continue
            # Consider directions the target can move to
            snake_dks = []
            safe_dks = []
            for dir in DIRECTIONS:
                pos_tmp = add_c(target_head_pos, dir)
                if (pos_tmp in WallPositions):
                    continue
                elif (pos_tmp in SnakeHitPositions):
                    snake_dks.append(dir2key(dir))
                else:
                    safe_dks.append(dir2key(dir))
            if (len(safe_dks) == 0):
                dks_to_cover = snake_dks
            else:
                dks_to_cover = safe_dks
            # If only one direction to cover
            if (len(dks_to_cover) == 1):
                target_pos = add_c(target_head_pos, key2dir(dks_to_cover[0]))
                keys_to_kill = search_path_keys(PositionHead, target_pos, WallPositions, poss_danger_rate=danger_map_for_kill)
                # If already at this position
                if (keys_to_kill == '' and next_max_lens[Num_] > 1):
                    kill_plans.append(r_key(dks_to_cover[0]))
                # If reachable, record
                if (keys_to_kill is not None and len(keys_to_kill) <= speed):
                    kill_plans.append(keys_to_kill)
                continue
            # Else get the path towards head
            keys_to_kill = search_path_keys(PositionHead, target_head_pos, WallPositions, poss_danger_rate=danger_map_for_kill)
            # If not reachable, ignore
            if (keys_to_kill is None or keys_to_kill == ''):
                continue
            # Else use dfs to search in 3*3 around enemy head
            poss_searched = set([PositionHead])
            pos_tmp = PositionHead
            for k in keys_to_kill[:-1]:
                pos_tmp = add_c(pos_tmp, key2dir(k))
                poss_searched.add(pos_tmp)
            keys_append = search_local_cover_path(
                target_head_pos,
                dks_to_cover,
                minus_c(target_head_pos, key2dir(keys_to_kill[-1])),
                WallPositions,
                poss_searched,
            )
            # If cannot cover any pos, ignore
            if (keys_append is None):
                continue
            keys_to_kill = keys_to_kill[:-1] + keys_append

            # # Old version (cross kill)
            # key_tmp = r_key(keys_to_kill[-1])
            # if (key_tmp in dks_to_cover):
            #     dks_to_cover.remove(key_tmp)
            # for dk in dks_to_cover:
            #     keys_to_kill += dk + r_key(dk)
            # keys_to_kill = keys_to_kill[:-1]

            # If reachable, record
            if (keys_to_kill is not None and len(keys_to_kill) <= speed and next_max_lens[Num_] >= 2*len(dks_to_cover)+1):
                kill_plans.append(keys_to_kill)
        # Calculate scores for kill plans (can be modified)
        if len(kill_plans) == 0:
            return None
        kill_plans_score = dict()
        for keys in kill_plans:
            # Danger rate (>=0)
            danger_rate = 0.0
            pos_tmp = PositionHead
            for k in keys:
                pos_tmp = add_c(pos_tmp, key2dir(k))
                if (pos_tmp in danger_map_for_kill):
                    danger_rate += danger_map_for_kill[pos_tmp]
            # Head part ratio
            head_part_ratio = min(len(keys) / next_max_lens[Num_], 1.0)
            # Score
            kill_plans_score[keys] = -danger_rate * head_part_ratio
        return max(kill_plans_score, key=kill_plans_score.get)
    
    kill_keys = get_manual_kill_keys()
    if kill_keys is not None and len(kill_keys) > 6:
        print(f'Round {game_map["Time"]}, Num_ = {Num_}, kill_keys = {kill_keys}')
        return kill_keys
    # Basic scores that doesnt depend on moving path
    pos_score_cache = dict()
    def get_pos_score(pos):
        if (pos in pos_score_cache):
            return pos_score_cache[pos]
        if (pos in WallPositions):
            if alive == 1 and rank_len == 6:
                return -params['score_hit_wall']
            # print('Hit wall:', pos)
            return params['score_hit_wall']
        res = 0
        for i_snake in range(len(players)):
            if (pos in SnakeHitPositions[i_snake]):
                if (i_snake == Num_ and is_strong):
                    res += params['score_hit_self_strong']
                elif (i_snake == Num_):
                    res += params['score_hit_wall']
                else:
                    res += params['score_hit_others_body']

        if (res < 0):
            return res
        if (pos in SugarPositions):
            alpha = params['score_sugar_alpha']
            beta = params['score_sugar_beta']
            gamma = params['score_sugar_gamma']
            if player_self["Score_time"] < 30:
                score_sugar = params['score_sugar']
            else:
                score_sugar = params['score_sugar_new']
            score_tmp = score_sugar * max(1, alpha - beta * player_self["Score_len"] - gamma * rank_len)
            if (is_double):
                score_tmp *= 2
            res += score_tmp
        elif (pos in SpeedPositions):
            alpha = params['score_speed_alpha']
            beta = params['score_speed_beta']
            gamma = params['score_speed_gamma']
            if player_self["Score_time"] < 30:
                score_speed = params['score_speed']
            else:
                score_speed = params['score_speed_new']
            res += score_speed * max(1, alpha - beta * speed - gamma * (1 - 1 / (1 + player_self["Prop"]["speed"])))
        elif (pos in StrongPositions):
            if player_self["Prop"]["strong"] > (150 - player_self["Score_time"]):
                res += params['score_strong_enough']
            else:
                alpha = params['score_strong_alpha']
                beta = params['score_strong_beta']
                gamma = params['score_strong_gamma']
                if player_self["Score_time"] < 30:
                    score_strong = params['score_strong']
                else:
                    score_strong = params['score_strong_new']
                res += score_strong * max(1, alpha - beta * is_strong - gamma * (1 - 1 / (1 + player_self["Prop"]["strong"])))
        elif (pos in DoublePositions):
            alpha = params['score_double_alpha']
            beta = params['score_double_beta']
            gamma = params['score_double_gamma']
            if player_self["Score_time"] < 30:
                score_double = params['score_double']
            else:
                score_double = params['score_double_new']
            res += score_double * max(1, alpha - beta * is_double - gamma * (1 - 1 / (1 + player_self["Prop"]["double"])))
        else:
            res += params['score_empty_grid']
        # Positions closer to center is better
        res += abs(pos[0] - (map_length-1)/2) * params['score_center_alpha_x']
        res += abs(pos[1] - (map_width-1)/2) * params['score_center_alpha_y']
        # score_map_enemy
        if (pos in score_map_enemy):
            res += score_map_enemy[pos]
        # According to distance of enemy
        for i_snake in range(player_num):
            if (power_levels[i_snake] == 'stronger'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_stronger_dist'] / dist_tmp
            elif (power_levels[i_snake] == 'weaker'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_weaker_dist'] / dist_tmp
            elif (power_levels[i_snake] == 'both_strong'):
                dist_tmp = ham_dist(pos, tuple(game_map["SnakePosition"][i_snake][0]))
                if (dist_tmp > 0):
                    res += params['score_enemy_both_strong_dist'] / dist_tmp

        res += params['score_constant']
        pos_score_cache[pos] = res        
        return res
    def get_connect_score(pos, poss):
        connect = 4
        for dk in DIRECTIONS_KEY:
            # if add_c(pos, key2dir(dk)) in (wall_and_snake_hit_poss | poss):
            pos_tmp = add_c(pos, key2dir(dk))
            if pos_tmp in wall_and_snake_hit_poss or pos_tmp in poss:
                connect -= 1
        if connect == 0:
            return params['score_connect_0']
        elif connect == 1:
            return params['score_connect_1']
        elif connect == 2:
            if max(200, (55 - player_self["Score_time"] / 2.5) * (40 - player_self["Score_time"] / 2.5)) / player_self["Score_len"] < 10:
                return params['score_connect_2'] + params['score_connect_2_new']
            else:
                return params['score_connect_2']
        elif connect == 3:
            return params['score_connect_3']
        print(connect)
        return 0

    max_search_step = 6
    hit_score_threshold = -10000
    act_values = dict()
    act_values_real = dict()        # For cases speed > length
    poss_cur = set([PositionHead])
    def iter_acts(act_cur, pos_cur, score_sum):
        if (len(act_cur) > 0):
            score_sum_cur = score_sum + get_pos_score(pos_cur) + get_connect_score(pos_cur, poss_cur)
            act_values[act_cur] = score_sum_cur
            # Consider cases speed > length, and add discount rate for future searches
            real_move_len = min(speed, len(act_cur))
            act_values_real[act_cur] = act_values[act_cur[:real_move_len]]
            for i in range(int((len(act_cur) - real_move_len) / speed)):
                act_values_real[act_cur] += pow(params['score_discount_rate'], i+1) * (act_values[act_cur[:real_move_len+speed*(i+1)]] - act_values[act_cur[:real_move_len+speed*i]])
            if (real_move_len > next_max_lens[Num_]):
                act_values_real[act_cur] -= act_values[act_cur[:real_move_len - next_max_lens[Num_]]]
        else:
            score_sum_cur = 0
        if (len(act_cur) < max_search_step and score_sum_cur > hit_score_threshold):
            for dk in DIRECTIONS_KEY:
                next_pos = add_c(pos_cur, key2dir(dk))
                if (next_pos not in poss_cur):
                    poss_cur.add(next_pos)
                    iter_acts(act_cur+dk, next_pos, score_sum_cur)
                    poss_cur.remove(next_pos)
    
    iter_acts('', PositionHead, 0)
    # print('Num =', Num_)
    if (debug):
        # print('act_values =', act_values)
        # print('act_values_real =', act_values_real)
        print('len(act_values_real) =', len(act_values_real))
        print('max([len(k) for k in act_values_real]) =', max([len(k) for k in act_values_real]))
    score_best = max(act_values_real.values())
    ok_keys = [dk for dk in act_values_real if act_values_real[dk] == score_best]
    if (debug):
        print('ok_keys =', ok_keys)
        print('score_best =', score_best)
    res_keys = random.choice(ok_keys)
    if (score_best > hit_score_threshold):
        # If there exists a path not hitting any snake body or wall
        if (debug):
            return res_keys
        else:
            return res_keys[:speed]

    
    if alive == 2:
        max_len = 0
        for i_snake in range(6):
            max_len = max(max_len, players[i_snake]["Score_len"])
            if players[i_snake]["IsDead"] != True and i_snake != Num_:
                enemy_strong = players[i_snake]["Prop"]["strong"]
        if params['w1'] * (max_len - players[Num_]["Score_len"]) + params['w2'] * (enemy_strong - players[Num_]["Prop"]["strong"]) + params['b'] > 0:
            return res_keys

    # Calculate whether can escape and decide whether to escape
    self_length_total = len(game_map["SnakePosition"][Num_]) + players[Num_]["SaveLength"]
    self_strong_time = players[Num_]["Prop"]["strong"]
    self_speed_time = players[Num_]["Prop"]["speed"]
    # If there is a wall formed by body of self, hit self until have a way out.
    extra_tol_len = 0   # should be at least 1 if self length > 1
    best_dk = None
    for dk in DIRECTIONS_KEY:
        dir = key2dir(dk)
        pos_tmp = add_c(PositionHead, dir)
        if (pos_tmp in SnakeHitPositions[Num_]):
            body_ind = game_map["SnakePosition"][Num_].index((pos_tmp[0], pos_tmp[1]))
            if (body_ind > extra_tol_len):
                extra_tol_len = body_ind
                best_dk = dk
    if (debug):
        print('extra_tol_len =', extra_tol_len)
    if (extra_tol_len < 1):
        return res_keys
    # 2-round speed sum
    valid_speed_time = min(self_speed_time, 2)
    two_speed_sum = speed * valid_speed_time + (2 - valid_speed_time)
    # max_speed_sum: speed sum in (self_strong_time + 1) rounds
    valid_speed_time = min(self_speed_time, self_strong_time + 1)
    max_speed_sum = speed * valid_speed_time + (self_strong_time + 1 - valid_speed_time)
    if (debug):
        print('self_length_total - extra_tol_len =', self_length_total - extra_tol_len)
        print('speed =', speed)
        print('two_speed_sum =', two_speed_sum)
        print('max_speed_sum =', max_speed_sum)
    # Three steps escape
    def hit_self_keys(first_key, num_step):
        res = first_key
        while (len(res) < num_step):
            res += r_key(res[-1])
        return res
    if (speed >= self_length_total - extra_tol_len and players[Num_]["SaveLength"] == 0):
        # SaveLength must be 0! otherwise would be complicated
        return best_dk + get_reverse_path_keys(game_map["SnakePosition"][Num_][extra_tol_len:])
    elif (speed >= self_length_total - extra_tol_len):
        return hit_self_keys(best_dk, players[Num_]["SaveLength"])
    elif (two_speed_sum >= self_length_total - extra_tol_len):
        len_to_destroy = self_length_total - extra_tol_len - speed  # should be >= 1
        return hit_self_keys(best_dk, len_to_destroy)
    elif (max_speed_sum >= self_length_total - extra_tol_len):
        return hit_self_keys(best_dk, speed)
    else:
        # cannot escape
        return res_keys








# def AI_greedy_wyg(Num_,GameInfo_):
#     player_self = GameInfo_["gameinfo"]["Player"][Num_]
#     if(player_self["IsDead"]):
#         return "d"

#     speed = player_self["Speed"]
#     PositionHead = tuple(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0])

#     WallPositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["WallPosition"]))
#     SugarPositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["SugarPosition"]))
#     SpeedPositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["PropPosition"][0]))
#     StrongPositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["PropPosition"][1]))
#     DoublePositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["PropPosition"][2]))
#     print(WallPositions)
#     SnakeHitPositions = []
#     for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
#         if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]):
#             SnakeHitPositions.append(set())
#         else:
#             # The last element of snake will not hit player
#             SnakeHitPositions.append(set(transfer_poss(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake])))

#     def get_pos_score(pos):
#         res = 0
#         if (pos in WallPositions):
#             res -= 10000
#         for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
#             if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]):
#                 continue
#             else:
#                 if (player_self["Prop"]["strong"] > 0 and GameInfo_["gameinfo"]["Player"][i_snake]["Prop"]["strong"] == 0) or len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]) > len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake]):
#                     res -= (abs(pos[0] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][0]) * 0.01 + abs(pos[1] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][1]) * 0.01)
#                     if abs(pos[0] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][0]) + abs(pos[1] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][1]) == 1:
#                         res += 10
#                 else:
#                     res += (abs(pos[0] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][0]) * 0.01 + abs(pos[1] - GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][0][1]) * 0.01)
#                 if (pos in SnakeHitPositions[i_snake]):
#                     if (i_snake == Num_ and player_self["Prop"]["strong"] > 0):
#                         res -= 1000
#                     else:
#                         res -= 20000
#         if (pos in SugarPositions):
#             if (player_self["Prop"]["double"] > 0):
#                 res += 2
#             else:
#                 res += 1
#         if (pos in SpeedPositions):
#             res += max(2, (5 - speed)) ** 2
#         if (pos in StrongPositions):
#             res += 5
#         if (pos in DoublePositions):
#             res += 2
#         res -= (abs(pos[0] - 27) * 0.1 + abs(pos[1] - 19.5) * 0.1)
#         return res

#     act_values = dict()
#     poss_cur = set([PositionHead])
#     def iter_acts(act_cur, pos_cur, score_sum, values):
#         if (len(act_cur) > 0):
#             score_sum_cur = score_sum + get_pos_score(pos_cur)
#             values[act_cur] = score_sum_cur
#         else:
#             score_sum_cur = 0
#         if (len(act_cur) < min(speed, 5) and score_sum_cur > -100):
#             for dk in DIRECTIONS_KEY:
#                 next_pos = add_c(pos_cur, key2dir(dk))
#                 if (next_pos not in poss_cur):
#                     poss_cur.add(next_pos)
#                     iter_acts(act_cur+dk, next_pos, score_sum_cur, values)
#                     poss_cur.remove(next_pos)
#     iter_acts('', PositionHead, 0, act_values)
#     HitPositions = WallPositions
#     for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
#         HitPositions = HitPositions | SnakeHitPositions[i_snake]
#     act_pos = PositionHead
#     def dfs(pos):
#         poss_cur.add(pos)
#         for dk in DIRECTIONS_KEY:
#             next_pos = add_c(pos, key2dir(dk))
#             if (next_pos not in (HitPositions | poss_cur)) and (abs(next_pos[0] - act_pos[0]) + abs(next_pos[1] - act_pos[1]) < 5):
#                 dfs(next_pos)
#     for dk in act_values:
#         next_act_values = dict()
#         act_pos = PositionHead
#         poss_cur = set([act_pos])
#         for i in range(len(dk)):
#             act_pos = add_c(act_pos, key2dir(dk[i]))
#             poss_cur.add(act_pos)
#         iter_acts('', act_pos, 0, next_act_values)
#         act_values[dk] += 0.99 * max(next_act_values.values())
#         dfs(act_pos)
#         act_values[dk] += (len(poss_cur) - len(dk))
#     score_best = max(act_values.values())
#     ok_keys = [dk for dk in act_values if act_values[dk] == score_best]
#     if (len(ok_keys) == 0):
#         return 'd'
#     else:
#         print(random.choice(ok_keys))
#         return random.choice(ok_keys)

# %%
