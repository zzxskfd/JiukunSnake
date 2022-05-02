# %%
import numpy as np
from datetime import datetime
import os
import random
import optuna
import gc
# from multiprocessing import Pool

from AI import AI0_rand, AI_greedy_0
from SnakeGame import SnakeGame
from utils import load_json, save_json

# python D:\zzx\Programming\vsCode\JiukunSnake\ParamTuning.py > D:\zzx\Programming\vsCode\JiukunSnake\tuning.log 2>&1
# %% Initial generations
# params = {
#     'score_hit_wall': -10000,
#     'score_hit_self_strong': -1000,
#     'score_hit_others_body': -30000,
#     'score_sugar': 1,
#     'score_speed': 2,
#     'score_strong': 5,
#     'score_double': 2,
#     'score_empty_grid': 0.1,
# }
# save_json(params, 'AI_greedy_0_params/params_0.json')

# %%
n_trials = 100
AI = AI_greedy_0
param_savepath_format = 'D:/zzx/Programming/vsCode/JiukunSnake/AI_greedy_0_params/params_{}.json'

# Find smallest unoccupied index
generation_ind = 0
while (os.path.exists(param_savepath_format.format(generation_ind))):
    generation_ind += 1

# %%
while generation_ind < 100:
    print('[%s] Start generation %d' % (datetime.now(), generation_ind))
    def objective(trial):
        params = {
            'score_hit_wall': trial.suggest_int('score_hit_wall', -100000, -100000),
            'score_hit_self_strong': trial.suggest_int('score_hit_self_strong', -150000, -150000),
            'score_hit_others_body': trial.suggest_int('score_hit_others_body', -200000, -200000),
            'score_sugar': trial.suggest_float('score_sugar', 1.0, 10.0),
            'score_sugar_alpha': trial.suggest_float('score_sugar_alpha', 5.0, 25.0),
            'score_sugar_beta': trial.suggest_float('score_sugar_beta', 0.0, 5.0),
            'score_sugar_gamma': trial.suggest_float('score_sugar_gamma', 0.0, 4.0),
            'score_speed': trial.suggest_float('score_speed', 1, 10),
            'score_speed_alpha': trial.suggest_float('score_speed_alpha', 5.0, 25.0),
            'score_speed_beta': trial.suggest_float('score_speed_beta', 0.0, 20.0),
            'score_speed_gamma': trial.suggest_float('score_speed_gamma', 0.0, 2.0),
            'score_strong': trial.suggest_float('score_strong', 1, 10),
            'score_strong_alpha': trial.suggest_float('score_strong_alpha', 5.0, 25.0),
            'score_strong_beta': trial.suggest_float('score_strong_beta', 0.0, 20.0),
            'score_strong_gamma': trial.suggest_float('score_strong_gamma', 0.0, 2.0),
            'score_double': trial.suggest_float('score_double', 1, 10),
            'score_double_alpha': trial.suggest_float('score_double_alpha', 5.0, 25.0),
            'score_double_beta': trial.suggest_float('score_double_beta', 0.0, 20.0),
            'score_double_gamma': trial.suggest_float('score_double_gamma', 0.0, 2.0),
            # 'score_empty_grid': trial.suggest_categorical('score_empty_grid', [0, 0.1, -0.1]),
            'score_empty_grid': trial.suggest_categorical('score_empty_grid', [0.0]),
            'score_center_alpha_x': trial.suggest_float('score_center_alpha_x', -10.0, 10.0),
            'score_center_alpha_y': trial.suggest_float('score_center_alpha_y', -10.0, 10.0),
            'score_enemy_stronger': trial.suggest_float('score_enemy_stronger', -50.0, 0.0),
            'score_enemy_weaker': trial.suggest_float('score_enemy_weaker', 0.0, 50.0),
            'score_enemy_both_strong': trial.suggest_float('score_enemy_both_strong', -20.0, 0.0),
            'score_enemy_stronger_dist': trial.suggest_float('score_enemy_stronger_dist', -50.0, 0.0),
            'score_enemy_weaker_dist': trial.suggest_float('score_enemy_weaker_dist', 0.0, 50.0),
            'score_enemy_both_strong_dist': trial.suggest_float('score_enemy_both_strong_dist', -10.0, 0.0),
            'score_discount_rate': trial.suggest_float('score_discount_rate', 0.0, 1.0),
        }

        def run_one_game():
            # Select random rivals for each game
            rival_inds = random.choices(
                population=list(range(generation_ind)),
                weights=[1, 1, 1, 1] + [2 * pow(1.2, i) for i in range(generation_ind-4)],
                k=5
            )
            rival_paramss = [load_json(param_savepath_format.format(ind)) for ind in rival_inds]
            
            def AI_tmp_0(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=params)
            def AI_tmp_1(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=rival_paramss[0])
            def AI_tmp_2(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=rival_paramss[1])
            def AI_tmp_3(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=rival_paramss[2])
            def AI_tmp_4(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=rival_paramss[3])
            def AI_tmp_5(Num_, GameInfo_):
                return AI(Num_, GameInfo_, params_=rival_paramss[4])

            game = SnakeGame(AIs=[AI_tmp_0, AI_tmp_1, AI_tmp_2, AI_tmp_3, AI_tmp_4, AI_tmp_5])
            game.run_till_end(print=False)
            scores = [game.players[i].Score for i in range(6)]
            return scores

        scores = []
        for i in range(400):
            if (i % 100 == 0):
                print('Game', i)
            scores.append(run_one_game())

        AI_scores = np.mean(scores, axis=0)
        print('AI_scores =', AI_scores, flush=True)
        return AI_scores[0]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    if (study.best_value > 3.5):
        save_json(study.best_params, param_savepath_format.format(generation_ind))
        generation_ind += 1
    else:
        break

    _ = gc.collect()



# %%
# def run_one_game():
    # game = SnakeGame(AIs=[AI_greedy_0 for _ in range(3)] + [AI0_rand for _ in range(3)])
    # game.run_till_end(print=False)
    # scores = [game.players[i].Score for i in range(6)]
    # return scores

# with Pool(processes=8) as pool:
#     scores = pool.starmap(run_one_game, [None for i in range(400)], chunksize=50)

# scores = []
# for i in range(100):
#     scores.append(run_one_game())


# %%
