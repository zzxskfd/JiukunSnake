# %%
import numpy as np
import random
from scipy.stats import rankdata
from copy import copy

from utils import DIRECTIONS, DIRECTIONS_KEY, get_reverse_path_keys, ham_dist, key2dir, add_c, r_key

def transfer_poss(poss):
    return [tuple(pos) for pos in poss]
    # return [(int(pos[0]), int(pos[1])) for pos in poss]

# %% AI example
def AI0(Num_,GameInfo_):
    #一个最简单的AI
    if(GameInfo_["gameinfo"]["Player"][Num_]["IsDead"]):
        return "w"
    #自身头部位置
    PositionNow = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    ActList = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}

    PositionMove = None
    for i in ActList:
        PositionMove = list(np.sum([PositionNow,ActList[i]],axis=0))
        #检查墙
        WallPosition_temp  = np.array(GameInfo_["gameinfo"]["Map"]["WallPosition"]).reshape(-1,2)
        if(((WallPosition_temp == PositionMove).sum(axis=1)==2).any()):#有墙
            #print(i,"wall")
            continue
        Hit = 0
        for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
            if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"] and (not GameInfo_["gameinfo"]["Player"][i_snake]["NowDead"])):
                continue
            if(len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake]) == 0):
                continue
            SnakePosition_temp  = np.array(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake]).reshape(-1,2)
            if(i == i_snake and np.sum((SnakePosition_temp == PositionMove).sum(axis=1)==2)>1):#判断重叠是否大于1
                #print(i,"snake")
                Hit = 1
                continue
            if(i != i_snake and np.sum((SnakePosition_temp == PositionMove).sum(axis=1)==2)>0):
                #print(i,"snake")
                Hit = 1
                continue
        if(Hit == 0):
            # print(PositionMove)
            return i
    # print(PositionMove)
    return "w"



# %% Dont kill self or hit wall
def AI0_rand(Num_, GameInfo_):
    if(GameInfo_["gameinfo"]["Player"][Num_]["IsDead"]):
        return "d"
    PositionNow = tuple(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0])

    PositionMove = None
    WallPositions  = set(transfer_poss(GameInfo_["gameinfo"]["Map"]["WallPosition"]))
    SnakeHitPositions = []
    for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
        if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
        else:
            # The last element of snake will not hit player
            SnakeHitPositions.append(set(transfer_poss(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake])))

    ok_flags = dict([(dk, True) for dk in DIRECTIONS_KEY])
    for dk in ok_flags:
        PositionMove = add_c(PositionNow, key2dir(dk))
        if(PositionMove in WallPositions):
            ok_flags[dk] = False
            continue
        for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
            if (PositionMove in SnakeHitPositions[i_snake]):
                ok_flags[dk] = False
                break

    ok_keys = [dk for dk in ok_flags if ok_flags[dk]]
    if (len(ok_keys) == 0):
        return 'd'
    else:
        return random.choice(ok_keys)
    



# %% Greedy for current round
def AI_greedy_0(Num_, GameInfo_, params_=None, debug=False):
    # Default params
    params = {
        'score_hit_wall': -10000,
        'score_hit_self_strong': -15000,
        'score_hit_others_body': -20000,
        'score_sugar': 1,
        'score_sugar_alpha': 1.0,   # >= 1
        'score_sugar_beta': 0.0,    # >= 0
        'score_sugar_gamma': 0.0,   # >= 0
        'score_speed': 2,
        'score_speed_alpha': 1.0,   # >= 1
        'score_speed_beta': 0.0,    # >= 0
        'score_speed_gamma': 0.0,   # >= 0
        'score_strong': 5,
        'score_strong_alpha': 1.0,   # >= 1
        'score_strong_beta': 0.0,    # >= 0
        'score_strong_gamma': 0.0,   # >= 0
        'score_double': 2,
        'score_double_alpha': 1.0,   # >= 1
        'score_double_beta': 0.0,    # >= 0
        'score_double_gamma': 0.0,   # >= 0
        'score_empty_grid': 0.1,
        'score_center_alpha_x': 0.0,    # if < 0, center is encouraged
        'score_center_alpha_y': 0.0,    # if < 0, center is encouraged
        'score_enemy_stronger': -0.0,       # should be negative
        'score_enemy_weaker': 0.0,          # should be positive
        'score_enemy_both_strong': -0.0,    # should be negative to discourage snake to move too much
        'score_enemy_stronger_dist': -0.0,      # should be negative
        'score_enemy_weaker_dist': 0.0,         # should be positive
        'score_enemy_both_strong_dist': -0.0,   # should be negative to discourage snake to move too much
        'score_discount_rate': 1.0,     # should be in [0.0, 1.0]
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
    for i_snake in range(player_num):
        if(players[i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
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

        pos_score_cache[pos] = res        
        return res

    # max_search_step = 12
    max_search_step = 6
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
            body_ind = game_map["SnakePosition"][Num_].index(pos_tmp)
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
    # Three steps escape
    if (speed >= self_length_total - extra_tol_len and players[Num_]["SaveLength"] == 0):
        # SaveLength must be 0! otherwise would be complicated
        return best_dk + get_reverse_path_keys(game_map["SnakePosition"][Num_][extra_tol_len:])
    elif (two_speed_sum >= self_length_total - extra_tol_len):
        len_to_destroy = self_length_total - extra_tol_len - speed  # should be >= 1
        keys_escape = best_dk
        while (len(keys_escape) < len_to_destroy):
            keys_escape += r_key(keys_escape[-1])
        return keys_escape
    elif (max_speed_sum >= self_length_total - extra_tol_len):
        keys_escape = best_dk
        while (len(keys_escape) < speed):
            keys_escape += r_key(keys_escape[-1])
        return keys_escape
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
