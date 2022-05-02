# %%
import numpy as np

from AI import AI_greedy_0
from SnakeGame import SnakeGame
from utils import load_json, save_json

# %%
n_trials = 100
AI = AI_greedy_0
param_savepath_format = 'D:/zzx/Programming/vsCode/JiukunSnake/AI_greedy_0_params/params_{}.json'
# %% Test tuned params
inds_test = [0, 1, 2, 5, 6, 7]
rival_paramss = [load_json(param_savepath_format.format(ind)) for ind in inds_test]

def AI_tmp_0(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[0])
def AI_tmp_1(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[1])
def AI_tmp_2(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[2])
def AI_tmp_3(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[3])
def AI_tmp_4(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[4])
def AI_tmp_5(Num_, GameInfo_):
    return AI(Num_, GameInfo_, params_=rival_paramss[5])

# %%
scores = []
scores_kill = []
scores_len = []
scores_time = []
for i in range(400):
    if (i % 100 == 0):
        print('Game', i)
    game = SnakeGame(AIs=[AI_tmp_0, AI_tmp_1, AI_tmp_2, AI_tmp_3, AI_tmp_4, AI_tmp_5])
    game.run_till_end(print=False)
    scores.append([game.players[i].Score for i in range(6)])
    scores_kill.append([game.players[i].Score_kill for i in range(6)])
    scores_len.append([game.players[i].Score_len for i in range(6)])
    scores_time.append([game.players[i].Score_time for i in range(6)])
print('Average Score =', np.mean(scores, axis=0))
print('Average Score_kill =', np.mean(scores_kill, axis=0))
print('Average Score_len =', np.mean(scores_len, axis=0))
print('Average Score_time =', np.mean(scores_time, axis=0))

# %%
game = SnakeGame(AIs=[AI_tmp_0, AI_tmp_1, AI_tmp_2, AI_tmp_3, AI_tmp_4, AI_tmp_5])
game.run_till_end(print=True, time_sleep=0.5, savedir='game_info_test')
print([game.players[i].Score_kill for i in range(6)])
print([game.players[i].Score_len for i in range(6)])
print([game.players[i].Score_time for i in range(6)])
print([game.players[i].Score for i in range(6)])

# %%
