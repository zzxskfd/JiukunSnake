import numpy as np

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
