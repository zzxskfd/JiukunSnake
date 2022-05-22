# %%
import numpy as np
import random
from scipy.stats import rankdata
from copy import copy

from utils import DIRECTIONS, DIRECTIONS_KEY, dir2key, get_reverse_path_keys, ham_dist, key2dir, add_c, minus_c, r_key, search_local_cover_path, search_path_keys, search_path_keys_multi_target

def transfer_poss(poss):
    return [tuple(pos) for pos in poss]
    # return [(int(pos[0]), int(pos[1])) for pos in poss]


def AI_reversing(Num_, GameInfoList_):
    GameInfo_ = GameInfoList_[-1]
    players = GameInfo_["gameinfo"]["Player"]
    player_self = players[Num_]
    if(player_self["IsDead"]):
        return "d"
    # cheating
    GameInfo_["gameinfo"]["Player"][Num_]["Prop"]["strong"] = 100
    self_poss = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
    if (len(self_poss) == 1):
        print(f'AI_reversing {Num_}, Round {GameInfo_["gameinfo"]["Map"]["Time"]}')
        return 'a'
    return dir2key(minus_c(self_poss[1], self_poss[0]))


def AI_20220520(Num_, GameInfoList_, params_=None, debug=False):
    # GameInfoList_: sorted by round (time)
    GameInfo_ = GameInfoList_[-1]
    # # [20220521] subtract 1 from buff durations
    # for player in GameInfo_["gameinfo"]["Player"]:
    #     player["Prop"]["speed"] = max(player["Prop"]["speed"] - 1, 0)
    #     player["Prop"]["strong"] = max(player["Prop"]["strong"] - 1, 0)
    #     player["Prop"]["double"] = max(player["Prop"]["double"] - 1, 0)
    # Default params
    params = {
        'score_hit_wall': -150000,
        'score_hit_self_strong': -100000,
        'score_hit_others_body': -200000,
        'score_hit_predicted_path': -90000,
        'score_sugar': 3.13,
        'score_sugar_alpha': 14.9,   # >= 1
        'score_sugar_beta': 1.73,    # >= 0
        'score_sugar_gamma': 0.48,   # >= 0
        'score_speed': 8.07,
        'score_speed_alpha': 18.21,   # >= 1
        'score_speed_beta': 0.29,    # >= 0
        'score_speed_gamma': 1.11,   # >= 0
        'score_strong': 6.28,
        'score_strong_alpha': 22.48,   # >= 1
        'score_strong_beta': 6.82,    # >= 0
        'score_strong_gamma': 0.92,   # >= 0
        'score_double': 7.97,
        'score_double_alpha': 10.3,   # >= 1
        'score_double_beta': 9.38,    # >= 0
        'score_double_gamma': 0.61,   # >= 0
        'score_empty_grid': 0,
        'score_center_alpha_x': -0.39,    # if < 0, center is encouraged
        'score_center_alpha_y': -0.39,    # if < 0, center is encouraged
        'score_enemy_stronger': -22.97,       # should be negative
        'score_enemy_weaker': 36.32,          # should be positive
        'score_enemy_both_strong': -12.72,    # should be negative to discourage snake to move too much
        'score_enemy_stronger_dist': -6.83,      # should be negative
        'score_enemy_weaker_dist': 29.77,         # should be positive
        'score_enemy_both_strong_dist': -2.15,   # should be negative to discourage snake to move too much
        'score_discount_rate': 0.74,     # should be in [0.0, 1.0]
        'score_constant': 74.28,
        'score_strong_enough': -6.72,
        'score_connect_0': -5.91,
        'score_connect_1': -5.72,
        'score_connect_2': 4.09,
        'score_connect_3': 6,
        'score_connect_2_new': 2.81,
        'score_straight': 1.46,
        'score_body': 7.26,
        'score_sugar_new': 3.03,
        'score_speed_new': 6.92,
        'score_strong_new': 11.22,
        'score_double_new': 4.57,
        'score_sugar_newer': 1.91,
        'score_speed_newer': 5.65,
        'score_strong_newer': 8.81,
        'score_double_newer': 5.98,
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

    # [20220510] Manual killer process (100% kill)
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
            # # Head part ratio
            # head_part_ratio = min(len(keys) / next_max_lens[Num_], 1.0)
            # Score
            # kill_plans_score[keys] = -danger_rate * head_part_ratio
            kill_plans_score[keys] = -len(keys) / next_max_lens[Num_] if danger_rate > 0 else 0.0
        res = max(kill_plans_score, key=kill_plans_score.get)
        if (len(res) <= 6):
            return None
        print(f'Round {game_map["Time"]}, Num_ = {Num_}, kill_keys = {res}, kill_plans_score[res] = {kill_plans_score[res]}')
        if (kill_plans_score[res] <= -0.999999):
            return None
        return res

    kill_keys = get_manual_kill_keys()
    if kill_keys is not None:
        return kill_keys

    # [20220520] Detect reversing pattern and hit (not 100% kill)
    def hit_reversing_snake_keys():
        if (not is_strong or next_max_lens[Num_] < 3):
            return None
        # Need at least 3 rounds
        if (len(GameInfoList_) < 3):
            return None
        snake_poss_0 = GameInfoList_[-1]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_1 = GameInfoList_[-2]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_2 = GameInfoList_[-3]["gameinfo"]["Map"]["SnakePosition"]
        hit_plans = dict()  # hit_keys -> player_num
        for i_snake in range(player_num):
            if (i_snake == Num_ or players[i_snake]["IsDead"]):
                continue
            if (len(snake_poss_0[i_snake]) < 2):
                continue
            if (not (snake_poss_0[i_snake] == snake_poss_1[i_snake][::-1] == snake_poss_2[i_snake])):
                continue
            # print(f'Round {game_map["Time"]}, Num_ = {Num_}, i_snake = {i_snake}')
            # Notice that enemy head should not be hit
            keys_to_hit = search_path_keys_multi_target(
                PositionHead, transfer_poss(snake_poss_0[i_snake][1:]), WallPositions, min(speed, next_max_lens[Num_]-1))
            if (keys_to_hit is not None):
                hit_plans[keys_to_hit] = i_snake
        
        if (len(hit_plans) == 0):
            return None
        hit_plans_score = dict()
        for keys in hit_plans:
            others_total_len = 0
            for i_snake in range(player_num):
                if (i_snake == Num_ or players[i_snake]["IsDead"]):
                    continue
                others_total_len += (players[i_snake]["Score_len"] + players[i_snake]["SaveLength"])
            self_total_len = players[Num_]["Score_len"] + players[Num_]["SaveLength"]
            # This is the benefit (erased enemy len / total other alive len (including saved length)) over loss (self erased len / self total len)
            score = ((players[hit_plans[keys]]["Score_len"] - 1) / others_total_len) / (len(keys) / self_total_len)
            hit_plans_score[keys] = score
        # print('hit_plans =', hit_plans)
        res = max(hit_plans_score, key=hit_plans_score.get)
        if (hit_plans_score[res] <= 1.0):
            return None
        else:
            print(f'Round {game_map["Time"]}, Num_ = {Num_}, hit_reversing_keys = {res}, target = {hit_plans[res]}')
            return res

    hit_reversing_keys = hit_reversing_snake_keys()
    if (hit_reversing_keys is not None):
        return hit_reversing_keys

    def hit_predicted_path_keys():
        if (not is_strong):
            return None, set()
        # Need at least 2 rounds
        if (len(GameInfoList_) < 2):
            return None, set()
        snake_poss_0 = GameInfoList_[-1]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_1 = GameInfoList_[-2]["gameinfo"]["Map"]["SnakePosition"]
        players_1 = GameInfoList_[-2]["gameinfo"]["Player"]
        # If not hit in previous round, ignore
        if (snake_poss_0[Num_][0] != snake_poss_1[Num_][0]):
            return None, set()
        # Calculate self's movement path
        last_act_tmp = players[Num_]["Act"]
        move_len_tmp = min(players_1[Num_]["Speed"], len(last_act_tmp))
        pos_tmp = tuple(snake_poss_1[Num_][0])
        poss_self_pre = set()
        for dk in last_act_tmp[:move_len_tmp]:
            pos_tmp = add_c(pos_tmp, key2dir(dk))
            poss_self_pre.add(pos_tmp)
        # Search for snakes that hit self
        hit_plans = dict()
        PredictedPathPositions = set()
        for i_snake in range(player_num):
            if (i_snake == Num_ or players[i_snake]["IsDead"]):
                continue
            if (snake_poss_0[i_snake][0] != snake_poss_1[i_snake][0]):
                continue
            # If speed different, ignore
            if (players[i_snake]["Speed"] != players_1[i_snake]["Speed"]):
                continue
            # If current length of enemy is short, ignore
            last_act_tmp = players[i_snake]["Act"]
            move_len_tmp = min(players_1[i_snake]["Speed"], len(last_act_tmp))
            if (next_max_lens[i_snake] <= move_len_tmp):
                continue
            # Calculate enemy's movement path
            pos_tmp = tuple(snake_poss_1[i_snake][0])
            poss_enemy_pre = set()
            for dk in last_act_tmp[:move_len_tmp]:
                pos_tmp = add_c(pos_tmp, key2dir(dk))
                poss_enemy_pre.add(pos_tmp)
            # If not hitting self, ignore
            if (poss_self_pre.isdisjoint(poss_enemy_pre)):
                continue
            PredictedPathPositions = PredictedPathPositions | poss_enemy_pre
            # Find shortest path to poss_enemy_pre
            keys_to_hit = search_path_keys_multi_target(
                PositionHead, poss_enemy_pre - {PositionHead}, WallPositions, min(speed, next_max_lens[Num_]-1))
            if (keys_to_hit is not None):
                hit_plans[keys_to_hit] = i_snake

        if (len(hit_plans) == 0):
            return None, set()
        # Consider plans
        hit_plans_score = dict()
        for keys in hit_plans:
            others_total_len = 0
            for i_snake in range(player_num):
                if (i_snake == Num_ or players[i_snake]["IsDead"]):
                    continue
                others_total_len += (players[i_snake]["Score_len"] + players[i_snake]["SaveLength"])
            self_total_len = players[Num_]["Score_len"] + players[Num_]["SaveLength"]
            # erased enemy len = move_len_tmp
            target = hit_plans[keys]
            last_act_tmp = players[target]["Act"]
            move_len_tmp = min(players_1[target]["Speed"], len(last_act_tmp))
            # This is the benefit (erased enemy len / total other alive len (including saved length)) over loss (self erased len / self total len)
            score = (move_len_tmp / others_total_len) / (len(keys) / self_total_len)
            hit_plans_score[keys] = score
        # print('hit_plans =', hit_plans)
        res = max(hit_plans_score, key=hit_plans_score.get)
        if (hit_plans_score[res] <= 1.0):
            return None, PredictedPathPositions
        else:
            # print(f'Round {game_map["Time"]}, Num_ = {Num_}, hit_predicted_path_keys = {res}, target = {hit_plans[res]}')
            return res, set()

    hit_path_keys, PredictedPathPositions = hit_predicted_path_keys()
    if (hit_path_keys is not None):
        return hit_path_keys

    # if (len(PredictedPathPositions) > 0):
    #     print(f'Round {game_map["Time"]}, Num_ = {Num_}, PredictedPathPositions = {PredictedPathPositions}')

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
        # [20220521] Predicted path
        if (pos in PredictedPathPositions):
            return params['score_hit_predicted_path']

        if (pos in SugarPositions):
            alpha = params['score_sugar_alpha']
            beta = params['score_sugar_beta']
            gamma = params['score_sugar_gamma']
            if player_self["Score_time"] < 30:
                score_sugar = params['score_sugar']
            elif player_self["Score_time"] < 80:
                score_sugar = params['score_sugar_new']
            else:
                score_sugar = params['score_sugar_newer']
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
            elif player_self["Score_time"] < 80:
                score_speed = params['score_speed_new']
            else:
                score_speed = params['score_speed_newer']
            res += score_speed * max(1, alpha - beta * speed - gamma * player_self["Prop"]["speed"])
        elif (pos in StrongPositions):
            if player_self["Prop"]["strong"] > (150 - player_self["Score_time"]):
                res += params['score_strong_enough']
            else:
                alpha = params['score_strong_alpha']
                beta = params['score_strong_beta']
                gamma = params['score_strong_gamma']
                if player_self["Score_time"] < 30:
                    score_strong = params['score_strong']
                elif player_self["Score_time"] < 80:
                    score_strong = params['score_strong_new']
                else:
                    score_strong = params['score_strong_newer']
                res += score_strong * max(1, alpha - beta * is_strong - gamma * player_self["Prop"]["strong"])
        elif (pos in DoublePositions):
            alpha = params['score_double_alpha']
            beta = params['score_double_beta']
            gamma = params['score_double_gamma']
            if player_self["Score_time"] < 30:
                score_double = params['score_double']
            elif player_self["Score_time"] < 80:
                score_double = params['score_double_new']
            else:
                score_double = params['score_double_newer']
            res += score_double * max(1, alpha - beta * is_double - gamma * player_self["Prop"]["double"])
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
            if add_c(pos, key2dir(dk)) in (wall_and_snake_hit_poss | poss):
                connect -= 1
        if connect == 0:
            return params['score_connect_0']
        elif connect == 1:
            return params['score_connect_1']
        elif connect == 2:
            if max(200, (55 - player_self["Score_time"] / 2.5) * (40 - player_self["Score_time"] / 2.5)) / max(player_self["Score_len"], 2) < 10:
                return params['score_connect_2'] + params['score_connect_2_new']
            else:
                return params['score_connect_2']
        elif connect == 3:
            return params['score_connect_3']
        print(connect)
        return 0

    # max_search_step = 12
    max_search_step = 5
    hit_score_threshold = -10000
    act_values = dict()
    act_values_real = dict()        # For cases speed > length
    poss_cur = set([PositionHead])
    def iter_acts(act_cur, pos_cur, score_sum):
        if (len(act_cur) > 0):
            score_sum_cur = score_sum + get_pos_score(pos_cur) + get_connect_score(pos_cur, poss_cur)
            if len(act_cur) > 1:
                if act_cur[-1] == act_cur[-2]:
                    score_sum_cur += params['score_straight']
            next_len = max(player_self["Score_len"], 2) + min(len(act_cur), player_self["SaveLength"])
            act_values[act_cur] = score_sum_cur + max((next_len - len(act_cur)) / next_len, 0) * params['score_body']
            # Consider cases speed > length, and add discount rate for future searches
            real_move_len = min(speed, len(act_cur))
            act_values_real[act_cur] = params['score_discount_rate'] * act_values[act_cur]\
                                       + (1-params['score_discount_rate']) * act_values[act_cur[:real_move_len]]
            if (real_move_len > next_max_lens[Num_]):
                act_values_real[act_cur] -= act_values[act_cur[:real_move_len - next_max_lens[Num_]]]
            # next_len = len(game_map["SnakePosition"][i_snake]) + min(len(act_cur), player_self["SaveLength"])
            # act_values_real[act_cur] += max((next_len - len(act_cur)) / next_len, 0) * params['score_body']
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


def AI_20220520_early(Num_, GameInfoList_, params_=None, debug=False):
    # GameInfoList_: sorted by round (time)
    GameInfo_ = GameInfoList_[-1]
    # # [20220521] subtract 1 from buff durations
    # for player in GameInfo_["gameinfo"]["Player"]:
    #     player["Prop"]["speed"] = max(player["Prop"]["speed"] - 1, 0)
    #     player["Prop"]["strong"] = max(player["Prop"]["strong"] - 1, 0)
    #     player["Prop"]["double"] = max(player["Prop"]["double"] - 1, 0)
    # Default params
    params = {
        'score_hit_wall': -150000,
        'score_hit_self_strong': -100000,
        'score_hit_others_body': -200000,
        'score_hit_predicted_path': -90000,
        'score_sugar': 3.13,
        'score_sugar_alpha': 14.9,   # >= 1
        'score_sugar_beta': 1.73,    # >= 0
        'score_sugar_gamma': 0.48,   # >= 0
        'score_speed': 8.07,
        'score_speed_alpha': 18.21,   # >= 1
        'score_speed_beta': 0.29,    # >= 0
        'score_speed_gamma': 1.11,   # >= 0
        'score_strong': 6.28,
        'score_strong_alpha': 22.48,   # >= 1
        'score_strong_beta': 6.82,    # >= 0
        'score_strong_gamma': 0.92,   # >= 0
        'score_double': 7.97,
        'score_double_alpha': 10.3,   # >= 1
        'score_double_beta': 9.38,    # >= 0
        'score_double_gamma': 0.61,   # >= 0
        'score_empty_grid': 0,
        'score_center_alpha_x': -0.39,    # if < 0, center is encouraged
        'score_center_alpha_y': -0.39,    # if < 0, center is encouraged
        'score_enemy_stronger': -22.97,       # should be negative
        'score_enemy_weaker': 36.32,          # should be positive
        'score_enemy_both_strong': -12.72,    # should be negative to discourage snake to move too much
        'score_enemy_stronger_dist': -6.83,      # should be negative
        'score_enemy_weaker_dist': 29.77,         # should be positive
        'score_enemy_both_strong_dist': -2.15,   # should be negative to discourage snake to move too much
        'score_discount_rate': 0.74,     # should be in [0.0, 1.0]
        'score_constant': 74.28,
        'score_strong_enough': -6.72,
        'score_connect_0': -5.91,
        'score_connect_1': -5.72,
        'score_connect_2': 4.09,
        'score_connect_3': 6,
        'score_connect_2_new': 2.81,
        'score_straight': 1.46,
        'score_body': 7.26,
        'score_sugar_new': 3.03,
        'score_speed_new': 6.92,
        'score_strong_new': 11.22,
        'score_double_new': 4.57,
        'score_sugar_newer': 1.91,
        'score_speed_newer': 5.65,
        'score_strong_newer': 8.81,
        'score_double_newer': 5.98,
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

    # [20220510] Manual killer process (100% kill)
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
        return kill_keys

    # [20220520] Detect reversing pattern and hit (not 100% kill)
    def hit_reversing_snake_keys():
        if (not is_strong or next_max_lens[Num_] < 3):
            return None
        # Need at least 3 rounds
        if (len(GameInfoList_) < 3):
            return None
        snake_poss_0 = GameInfoList_[-1]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_1 = GameInfoList_[-2]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_2 = GameInfoList_[-3]["gameinfo"]["Map"]["SnakePosition"]
        hit_plans = dict()  # hit_keys -> player_num
        for i_snake in range(player_num):
            if (i_snake == Num_ or players[i_snake]["IsDead"]):
                continue
            if (len(snake_poss_0[i_snake]) < 2):
                continue
            if (not (snake_poss_0[i_snake] == snake_poss_1[i_snake][::-1] == snake_poss_2[i_snake])):
                continue
            # print(f'Round {game_map["Time"]}, Num_ = {Num_}, i_snake = {i_snake}')
            # Notice that enemy head should not be hit
            keys_to_hit = search_path_keys_multi_target(
                PositionHead, transfer_poss(snake_poss_0[i_snake][1:]), WallPositions, min(speed, next_max_lens[Num_]-1))
            if (keys_to_hit is not None):
                hit_plans[keys_to_hit] = i_snake
        
        if (len(hit_plans) == 0):
            return None
        hit_plans_score = dict()
        for keys in hit_plans:
            others_total_len = 0
            for i_snake in range(player_num):
                if (i_snake == Num_ or players[i_snake]["IsDead"]):
                    continue
                others_total_len += (players[i_snake]["Score_len"] + players[i_snake]["SaveLength"])
            self_total_len = players[Num_]["Score_len"] + players[Num_]["SaveLength"]
            # This is the benefit (erased enemy len / total other alive len (including saved length)) over loss (self erased len / self total len)
            score = ((players[hit_plans[keys]]["Score_len"] - 1) / others_total_len) / (len(keys) / self_total_len)
            hit_plans_score[keys] = score
        # print('hit_plans =', hit_plans)
        res = max(hit_plans_score, key=hit_plans_score.get)
        if (hit_plans_score[res] <= 1.0):
            return None
        else:
            print(f'Round {game_map["Time"]}, Num_ = {Num_}, hit_reversing_keys = {res}, target = {hit_plans[res]}')
            return res

    hit_reversing_keys = hit_reversing_snake_keys()
    if (hit_reversing_keys is not None):
        return hit_reversing_keys

    def hit_predicted_path_keys():
        if (not is_strong):
            return None, set()
        # Need at least 2 rounds
        if (len(GameInfoList_) < 2):
            return None, set()
        snake_poss_0 = GameInfoList_[-1]["gameinfo"]["Map"]["SnakePosition"]
        snake_poss_1 = GameInfoList_[-2]["gameinfo"]["Map"]["SnakePosition"]
        players_1 = GameInfoList_[-2]["gameinfo"]["Player"]
        # If not hit in previous round, ignore
        if (snake_poss_0[Num_][0] != snake_poss_1[Num_][0]):
            return None, set()
        # Calculate self's movement path
        last_act_tmp = players[Num_]["Act"]
        move_len_tmp = min(players_1[Num_]["Speed"], len(last_act_tmp))
        pos_tmp = tuple(snake_poss_1[Num_][0])
        poss_self_pre = set()
        for dk in last_act_tmp[:move_len_tmp]:
            pos_tmp = add_c(pos_tmp, key2dir(dk))
            poss_self_pre.add(pos_tmp)
        # Search for snakes that hit self
        hit_plans = dict()
        PredictedPathPositions = set()
        for i_snake in range(player_num):
            if (i_snake == Num_ or players[i_snake]["IsDead"]):
                continue
            if (snake_poss_0[i_snake][0] != snake_poss_1[i_snake][0]):
                continue
            # If speed different, ignore
            if (players[i_snake]["Speed"] != players_1[i_snake]["Speed"]):
                continue
            # If current length of enemy is short, ignore
            last_act_tmp = players[i_snake]["Act"]
            move_len_tmp = min(players_1[i_snake]["Speed"], len(last_act_tmp))
            if (next_max_lens[i_snake] <= move_len_tmp):
                continue
            # Calculate enemy's movement path
            pos_tmp = tuple(snake_poss_1[i_snake][0])
            poss_enemy_pre = set()
            for dk in last_act_tmp[:move_len_tmp]:
                pos_tmp = add_c(pos_tmp, key2dir(dk))
                poss_enemy_pre.add(pos_tmp)
            # If not hitting self, ignore
            if (poss_self_pre.isdisjoint(poss_enemy_pre)):
                continue
            PredictedPathPositions = PredictedPathPositions | poss_enemy_pre
            # Find shortest path to poss_enemy_pre
            keys_to_hit = search_path_keys_multi_target(
                PositionHead, poss_enemy_pre - {PositionHead}, WallPositions, min(speed, next_max_lens[Num_]-1))
            if (keys_to_hit is not None):
                hit_plans[keys_to_hit] = i_snake

        if (len(hit_plans) == 0):
            return None, set()
        # Consider plans
        hit_plans_score = dict()
        for keys in hit_plans:
            others_total_len = 0
            for i_snake in range(player_num):
                if (i_snake == Num_ or players[i_snake]["IsDead"]):
                    continue
                others_total_len += (players[i_snake]["Score_len"] + players[i_snake]["SaveLength"])
            self_total_len = players[Num_]["Score_len"] + players[Num_]["SaveLength"]
            # erased enemy len = move_len_tmp
            target = hit_plans[keys]
            last_act_tmp = players[target]["Act"]
            move_len_tmp = min(players_1[target]["Speed"], len(last_act_tmp))
            # This is the benefit (erased enemy len / total other alive len (including saved length)) over loss (self erased len / self total len)
            score = (move_len_tmp / others_total_len) / (len(keys) / self_total_len)
            hit_plans_score[keys] = score
        # print('hit_plans =', hit_plans)
        res = max(hit_plans_score, key=hit_plans_score.get)
        if (hit_plans_score[res] <= 1.0):
            return None, PredictedPathPositions
        else:
            # print(f'Round {game_map["Time"]}, Num_ = {Num_}, hit_predicted_path_keys = {res}, target = {hit_plans[res]}')
            return res, set()

    # hit_path_keys, PredictedPathPositions = hit_predicted_path_keys()
    # if (hit_path_keys is not None):
    #     return hit_path_keys

    # if (len(PredictedPathPositions) > 0):
    #     print(f'Round {game_map["Time"]}, Num_ = {Num_}, PredictedPathPositions = {PredictedPathPositions}')

    PredictedPathPositions = set()

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
        # [20220521] Predicted path
        if (pos in PredictedPathPositions):
            return params['score_hit_predicted_path']

        if (pos in SugarPositions):
            alpha = params['score_sugar_alpha']
            beta = params['score_sugar_beta']
            gamma = params['score_sugar_gamma']
            if player_self["Score_time"] < 30:
                score_sugar = params['score_sugar']
            elif player_self["Score_time"] < 80:
                score_sugar = params['score_sugar_new']
            else:
                score_sugar = params['score_sugar_newer']
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
            elif player_self["Score_time"] < 80:
                score_speed = params['score_speed_new']
            else:
                score_speed = params['score_speed_newer']
            res += score_speed * max(1, alpha - beta * speed - gamma * player_self["Prop"]["speed"])
        elif (pos in StrongPositions):
            if player_self["Prop"]["strong"] > (150 - player_self["Score_time"]):
                res += params['score_strong_enough']
            else:
                alpha = params['score_strong_alpha']
                beta = params['score_strong_beta']
                gamma = params['score_strong_gamma']
                if player_self["Score_time"] < 30:
                    score_strong = params['score_strong']
                elif player_self["Score_time"] < 80:
                    score_strong = params['score_strong_new']
                else:
                    score_strong = params['score_strong_newer']
                res += score_strong * max(1, alpha - beta * is_strong - gamma * player_self["Prop"]["strong"])
        elif (pos in DoublePositions):
            alpha = params['score_double_alpha']
            beta = params['score_double_beta']
            gamma = params['score_double_gamma']
            if player_self["Score_time"] < 30:
                score_double = params['score_double']
            elif player_self["Score_time"] < 80:
                score_double = params['score_double_new']
            else:
                score_double = params['score_double_newer']
            res += score_double * max(1, alpha - beta * is_double - gamma * player_self["Prop"]["double"])
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
            if add_c(pos, key2dir(dk)) in (wall_and_snake_hit_poss | poss):
                connect -= 1
        if connect == 0:
            return params['score_connect_0']
        elif connect == 1:
            return params['score_connect_1']
        elif connect == 2:
            if max(200, (55 - player_self["Score_time"] / 2.5) * (40 - player_self["Score_time"] / 2.5)) / max(player_self["Score_len"], 2) < 10:
                return params['score_connect_2'] + params['score_connect_2_new']
            else:
                return params['score_connect_2']
        elif connect == 3:
            return params['score_connect_3']
        print(connect)
        return 0

    # max_search_step = 12
    max_search_step = 5
    hit_score_threshold = -10000
    act_values = dict()
    act_values_real = dict()        # For cases speed > length
    poss_cur = set([PositionHead])
    def iter_acts(act_cur, pos_cur, score_sum):
        if (len(act_cur) > 0):
            score_sum_cur = score_sum + get_pos_score(pos_cur) + get_connect_score(pos_cur, poss_cur)
            if len(act_cur) > 1:
                if act_cur[-1] == act_cur[-2]:
                    score_sum_cur += params['score_straight']
            next_len = max(player_self["Score_len"], 2) + min(len(act_cur), player_self["SaveLength"])
            act_values[act_cur] = score_sum_cur + max((next_len - len(act_cur)) / next_len, 0) * params['score_body']
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

