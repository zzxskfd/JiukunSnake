# %%
import numpy as np
import random

from utils import DIRECTIONS_KEY, key2dir, add_c

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
    PositionNow = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]

    PositionMove = None
    WallPositions  = set(GameInfo_["gameinfo"]["Map"]["WallPosition"])
    SnakeHitPositions = []
    for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
        if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
        else:
            # The last element of snake will not hit player
            SnakeHitPositions.append(set((GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][:-1])))

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
def AI_greedy_0(Num_, GameInfo_):
    player_self = GameInfo_["gameinfo"]["Player"][Num_]
    if(player_self["IsDead"]):
        return "d"

    speed = player_self["Speed"]
    PositionHead = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]

    WallPositions  = set(GameInfo_["gameinfo"]["Map"]["WallPosition"])
    SugarPositions  = set(GameInfo_["gameinfo"]["Map"]["SugarPosition"])
    SpeedPositions  = set(GameInfo_["gameinfo"]["Map"]["PropPosition"][0])
    StrongPositions  = set(GameInfo_["gameinfo"]["Map"]["PropPosition"][1])
    DoublePositions  = set(GameInfo_["gameinfo"]["Map"]["PropPosition"][2])
    SnakeHitPositions = []
    for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
        if(GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]):
            SnakeHitPositions.append(set())
        else:
            # The last element of snake will not hit player
            SnakeHitPositions.append(set((GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake][:-1])))

    def get_pos_score(pos):
        if (pos in WallPositions):
            return -10000
        res = 0
        for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
            if (pos in SnakeHitPositions[i_snake]):
                if (i_snake == Num_ and player_self["Prop"]["strong"] > 0):
                    res -= 1000
                else:
                    res -= 20000
        if (res < 0):
            return res
        if (pos in SugarPositions):
            if (player_self["Prop"]["double"] > 0):
                return 2
            else:
                return 1
        if (pos in SpeedPositions):
            return max(2, (4 - speed) ** 2)
        if (pos in StrongPositions):
            return 5
        if (pos in DoublePositions):
            return 2
        return 0.1

    act_values = dict()
    poss_cur = set([PositionHead])
    def iter_acts(act_cur, pos_cur, score_sum):
        if (len(act_cur) > 0):
            score_sum_cur = score_sum + get_pos_score(pos_cur)
            act_values[act_cur] = score_sum_cur
        else:
            score_sum_cur = 0
        if (len(act_cur) < min(speed, 4) and score_sum_cur > -100):
            for dk in DIRECTIONS_KEY:
                next_pos = add_c(pos_cur, key2dir(dk))
                if (next_pos not in poss_cur):
                    poss_cur.add(next_pos)
                    iter_acts(act_cur+dk, next_pos, score_sum_cur)
                    poss_cur.remove(next_pos)
    
    iter_acts('', PositionHead, 0)
    print('Num =', Num_)
    print('act_values =', act_values)
    score_best = max(act_values.values())
    ok_keys = [dk for dk in act_values if act_values[dk] == score_best]
    print('ok_keys =', ok_keys)
    if (len(ok_keys) == 0):
        return 'd'
    else:
        return random.choice(ok_keys)
    

