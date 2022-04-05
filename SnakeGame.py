# %%
import numpy as np
import gc
import os
import random
from scipy.stats import rankdata
from copy import deepcopy
from time import sleep
from colorama import Fore, Style
from multiprocessing import Pool

from AI0 import AI0
from utils import create_folder, save_json, load_json

# %%
RIGHT = (0, 1)
LEFT = (0, -1)
UP = (-1, 0)
DOWN = (1, 0)

DIRECTIONS = [RIGHT, LEFT, UP, DOWN]

def key2dir(key: str):
    if (key == 'w'):
        return UP
    if (key == 'a'):
        return LEFT
    if (key == 's'):
        return DOWN
    if (key == 'd'):
        return RIGHT
    raise ValueError(f'Error: invalid key: {key}')

def add_c(x: tuple, y: tuple):
    return (x[0]+y[0], x[1]+y[1])


# %%
class Player:
    def __init__(self, name, num) -> None:
        self.Name = name
        self.Num = num
        self.LastAct = 'w'
        # self.Act = 'w'
        self.IsDead = False
        self.NowDead = False
        self.Speed = 1
        self.Kill = 0
        self.KillList = []
        self.KilledList = []
        self.SaveLength = 0
        self.Prop = {
            'speed': 0,
            'strong': 0,
            'double': 0,
        }
        # self.del_act = None
        self.Score_len = None
        self.Score_kill = None
        self.Score_time = None
        self.Score = None
        # Addtional params
        self.move_len = 0
        self.total_len = 0
        self.remove_head = False


# %%
class Map:
    def __init__(self, num_players, max_time=150) -> None:
        self.Length = 55    # X
        self.Width = 40     # Y
        # TODO Snake initial positions for 4/5 players? Use random when training?
        self.SnakePosition = [[] for _ in range(num_players)]
        if (num_players == 6):
            x0 = int(self.Length / 4)
            y0 = int(self.Width / 3)
            init_pos = [
                [x0, y0], [2 * x0, y0], [3 * x0, y0],
                [x0, y0 * 2], [2 * x0, y0 * 2], [3 * x0, y0 * 2],
            ]
            init_pos = [tuple(p) for p in init_pos]
            random.shuffle(init_pos)
            for i, pos in enumerate(init_pos):
                self.SnakePosition[i] = [pos, add_c(pos, random.choice(DIRECTIONS))]

        self.SugarPosition = set()
        self.WallPosition = set()
        self.PropPosition = [set(), set(), set()]   # speed, strong, double
        self.Time = 0
        self.Score = [None for _ in range(num_players)]
        # Additional parameters
        self.max_time = max_time
        self.wall_gen_ind = 0
        self.all_grids = set([(x, y) for x in range(self.Length) for y in range(self.Width)])
    
    def gen_walls_and_props(self):
        # TODO
        # Generate walls
        if (self.Time % 5 == 0 and (self.Time >= 100 or self.Length * self.Width - len(self.WallPosition) > 400)):
            i = self.wall_gen_ind
            for j in range(self.Width):
                self.WallPosition.add((i, j))
                self.WallPosition.add((self.Length - 1 - i, j))
            for j in range(self.Length):
                self.WallPosition.add((j, i))
                self.WallPosition.add((j, self.Width - 1 - i))
            self.WallPosition = self.vacant_subset(self.WallPosition, wall=False)
            self.wall_gen_ind += 1
        # Generate sugars and props
        if (self.Time == 0):
            # Initial num: 200 sugar, 50/50/50 props
            sugar_num = 200
            prop_num = 50
            poss = self.__gen_gauss_poss__(sugar_num + 3 * prop_num, range=self.vacant_subset())
            for i in range(3):
                self.PropPosition[i] = set(poss[i*prop_num:(i+1)*prop_num])
            self.SugarPosition = set(poss[3*prop_num:])
        if (self.Time == 1):
            # When time == 1, generate extra 10 speed props
            poss = self.__gen_gauss_poss__(10, range=self.vacant_subset())
            self.PropPosition[0] = self.PropPosition[0].union(poss)
        # Sugar
        sugar_num = random.choice([1, 1, 1, 1, 2, 2, 3, 3, 4])
        poss = self.__gen_gauss_poss__(sugar_num)
        poss = self.vacant_subset(poss)
        self.SugarPosition = self.SugarPosition.union(poss)
        # Props
        for i in [0, 2, 1]:
            # speed > double > strong
            poss = self.__gen_gauss_poss__(1)
            poss = self.vacant_subset(poss)
            self.PropPosition[i] = self.PropPosition[i].union(poss)

    def __gen_gauss_poss__(self, num, range:set=None):
        mu_x = (self.Length - 1) / 2
        mu_y = (self.Width - 1) / 2
        sigma_x = self.Length / 2
        sigma_y = self.Width / 2
        # if not enough spaces, return all remaining spaces
        if (range is not None and num >= len(range)):
            poss = range
            poss = list(poss)
            random.shuffle(poss)
            return poss
        poss = set()
        while (len(poss) < num):
            pos = (int(np.round(random.gauss(mu_x, sigma_x))),
                    int(np.round(random.gauss(mu_y, sigma_y))))
            while (not self.is_valid_pos(pos) or (range is not None and not pos in range)):
                pos = (int(np.round(random.gauss(mu_x, sigma_x))),
                    int(np.round(random.gauss(mu_y, sigma_y))))
            poss.add(pos)
        poss = list(poss)
        random.shuffle(poss)
        return poss

    def time_over(self):
        return self.Time >= self.max_time
    
    def is_valid_pos(self, pos):
        return (0 <= pos[0] <= self.Length - 1) and (0 <= pos[1] <= self.Width - 1)

    def is_valid_poss(self, poss):
        return np.all([self.is_valid_pos(pos) for pos in poss])
    
    def vacant_subset(self, poss=None, wall=True, sugar=True, prop=True, snake=True) -> set:
        if (poss is None):
            poss = self.all_grids
        res = poss.copy()
        if (type(res) != set):
            res = set(res)
        if (snake):
            for snake_poss in self.SnakePosition:
                res -= set(snake_poss)
        if (sugar):
            res -= self.SugarPosition
        if (wall):
            res -= self.WallPosition
        if (prop):
            for i in range(3):
                res -= self.PropPosition[i]
        return res
    
    def load_game_info(self, loadpath, as_set=True):
        game_info = load_json(loadpath)
        self.__dict__.update(game_info['gameinfo']['Map'])
        if (as_set):
            # convert data to set of tuples
            self.SugarPosition = set([tuple(pos) for pos in self.SugarPosition])
            self.WallPosition = set([tuple(pos) for pos in self.WallPosition])
            for i in range(3):
                self.PropPosition[i] = set([tuple(pos) for pos in self.PropPosition[i]])

    def print(self, wall=True, sugar=True, prop=True, snake=True):
        wall_char = 'ww'
        sugar_char = '* '
        prop_chars = ['sp', 'st', 'do']   # speed, strong, double
        # prop_chars = [Fore.LIGHTBLUE_EX + c + Style.RESET_ALL for c in prop_chars]
        player_colors = [
            Fore.RED,
            Fore.GREEN,
            Fore.BLUE,
            Fore.YELLOW,
            Fore.CYAN,
            Fore.MAGENTA,
        ]
        snake_head_chars = [player_colors[i]+'@@'+Style.RESET_ALL for i in range(len(self.SnakePosition))]
        snake_body_chars = [player_colors[i]+'OO'+Style.RESET_ALL for i in range(len(self.SnakePosition))]

        board = [['  ' for i in range(self.Width)] for j in range(self.Length)]
        if (wall):
            for pos in self.WallPosition:
                # # Ignore the outer walls
                # if (not self.is_valid_pos(pos)):
                #     continue
                # if (board[pos[0]][pos[1]] != '  '):
                #     print(f'Warning: conflict on {pos[0]}{pos[1]}')
                board[pos[0]][pos[1]] = wall_char
        if (sugar):
            for pos in self.SugarPosition:
                board[pos[0]][pos[1]] = sugar_char
        if (prop):
            for i in range(3):
                for pos in self.PropPosition[i]:
                    board[pos[0]][pos[1]] = prop_chars[i]
        if (snake):
            for i, positions in enumerate(self.SnakePosition):
                if (len(positions) == 0):
                    continue
                pos = positions[0]
                board[pos[0]][pos[1]] = snake_head_chars[i]
                for pos in positions[1:]:
                    board[pos[0]][pos[1]] = snake_body_chars[i]
        board = np.transpose(board)
        print('\n'.join([''.join(row) for row in board]))
        

# %%
class SnakeGame:
    def __init__(self, AIs=None) -> None:
        self.reset(AIs=AIs)

    def reset(self, AIs=None):
        self.player_num = len(AIs)
        self.AIs = AIs
        self.players = [Player(f'player_{i}', i) for i in range(self.player_num)]
        self.map = Map(self.player_num)
        self.map.gen_walls_and_props()
        self.__calc_scores__()

    def get_game_info(self, to_list=True):
        game_info = dict()
        game_info['Player'] = []
        for p in self.players:
            tmp = deepcopy(p.__dict__)
            tmp.pop('move_len')
            tmp.pop('total_len')
            tmp.pop('remove_head')
            game_info['Player'].append(tmp)
        game_info['Map'] = deepcopy(self.map.__dict__)
        game_info['Map'].pop('all_grids')
        if (to_list):
            game_info['Map']['WallPosition'] = list(game_info['Map']['WallPosition'])
            game_info['Map']['SugarPosition'] = list(game_info['Map']['SugarPosition'])
            for i in range(3):
                game_info['Map']['PropPosition'][i] = list(game_info['Map']['PropPosition'][i])

        res = dict()
        res['gameinfo'] = game_info
        res['tableinfo'] = None
        return res

    # def save_game(self, save_path):
    #     config = deepcopy(self.__dict__)
    #     config['AIs'] = None
    #     save_json(config, save_path)
    
    # def load_game(self, load_path):
    #     self.reset()
    #     json = load_json(load_path)

    def print(self, cls=True):
        if (cls):
            os.system('cls')
        print(f'Round {self.map.Time}')
        self.map.print()

    def run_till_end(self, time_sleep=0, savedir=None):
        # Run a whole game
        if (savedir is not None):
            create_folder(savedir)
        def print_and_save():
            self.print()
            if (savedir is not None):
                save_json(
                    self.get_game_info(to_list=True),
                    os.path.join(savedir, 'game_info_round_%3d.json' % (self.map.Time))
                )
    
        print_and_save()
        sleep(time_sleep)
        while (self.move_one_round()):
            print_and_save()
            sleep(time_sleep)
        print_and_save()
    
    def move_one_round(self, AIs=None, acts=None) -> bool:
        _ = gc.collect()
        # Return true if game is not ended
        if (acts is None):
            if (AIs is None):
                AIs = self.AIs
            game_info = self.get_game_info()
            acts = self.__ask_for_acts__(game_info)
        self.__move_players__(acts)
        self.__collide__()
        self.__update_time__()
        self.__get_props__()
        self.__confirm_new_snakes__()
        self.__calc_scores__()
        self.map.gen_walls_and_props()
        if (self.__check_all_dead__()):
            return False
        if (self.map.time_over()):
            self.__kill_all__()
            return False
        return True
    
    def __ask_for_acts__(self, game_info):
        acts = []
        for i, AI in enumerate(self.AIs):
            acts.append(AI(i, game_info))
        # with Pool(processes=8) as pool:
        #     acts = pool.starmap(get_act, list(range(self.player_num)))
        return acts
    
    def __move_players__(self, acts):
        # if act is None, use last act
        for i, act in enumerate(acts):
            if (act is None):
                act = self.players[i].LastAct
        for i in range(self.player_num):
            # Ignore previously dead snakes
            if (self.players[i].IsDead):
                continue
            self.players[i].move_len = 0
            self.players[i].total_len = len(self.map.SnakePosition[i])
            # Can move up to 'speed' times
            max_step = self.players[i].Speed
            act = acts[i]
            act = act[:max_step]
            for key in act:
                dir = key2dir(key)
                assert dir is not None
                self.map.SnakePosition[i].insert(0, add_c(self.map.SnakePosition[i][0], dir))
                self.players[i].move_len += 1
        # Set last_Act
        for i in range(self.player_num):
            self.players[i].LastAct = acts[i]
    
    def __collide__(self):
        # # Initialize now_dead
        # for i in range(self.player_num):
        #     self.players[i].now_dead = self.players[i].is_dead

        head_poss = []
        body_poss = []
        is_strong = []
        for i in range(self.player_num):
            positions = self.map.SnakePosition[i]
            head_len = self.players[i].move_len
            total_len = self.players[i].total_len
            head_poss.append(set(positions[:head_len]))
            body_poss.append(set(positions[head_len:total_len]))
            is_strong.append(self.players[i].Prop['strong'] > 0)

        for i in range(self.player_num):
            # Ignore previously dead snakes
            if (self.players[i].IsDead):
                continue
            # Collide with walls
            if (not self.map.is_valid_poss(head_poss[i]) or not head_poss[i].isdisjoint(self.map.WallPosition)):
                self.players[i].NowDead = True
            # Collide with self
            if (not head_poss[i].isdisjoint(body_poss[i])):
                if (is_strong[i]):
                    self.players[i].remove_head = True
                else:
                    self.players[i].NowDead = True
            # Collide with others
            for j in range(i+1, self.player_num):
                collide_h2h = not head_poss[i].isdisjoint(head_poss[j])
                collide_h2b = not head_poss[i].isdisjoint(body_poss[j])
                collide_b2h = not body_poss[i].isdisjoint(head_poss[j])
                if (collide_h2h):
                    if (collide_h2b):
                        if (is_strong[i]):
                            self.players[i].remove_head = True
                        else:
                            self.__on_kill__(j, i, True)
                    if (collide_b2h):
                        if (is_strong[j]):
                            self.players[j].remove_head = True
                        else:
                            self.__on_kill__(i, j, True)
                    if (not collide_h2b and not collide_b2h):
                        if (not is_strong[i] and not is_strong[j]):
                            if (self.players[i].total_len > self.players[j].total_len):
                                self.__on_kill__(i, j, True)
                            elif (self.players[i].total_len < self.players[j].total_len):
                                self.__on_kill__(j, i, True)
                            else:
                                self.players[i].NowDead = True
                                self.players[j].NowDead = True
                        elif (is_strong[i] and not is_strong[j]):
                            self.__on_kill__(i, j, True)
                        elif (not is_strong[i] and is_strong[j]):
                            self.__on_kill__(j, i, True)
                        else:
                            self.players[i].remove_head = True
                            self.players[j].remove_head = True
                else:
                    if (collide_h2b):
                        if (is_strong[i]):
                            self.players[i].remove_head = True
                        else:
                            self.__on_kill__(j, i, False)
                    if (collide_b2h):
                        if (is_strong[j]):
                            self.players[j].remove_head = True
                        else:
                            self.__on_kill__(i, j, False)

        # Realize remove head
        for i in range(self.player_num):
            if (self.players[i].remove_head):
                assert(not self.players[i].NowDead)
                self.__remove_head__(i)

    def __on_kill__(self, killer: int, killed: int, active_kill: bool):
        self.players[killed].NowDead = True
        self.players[killed].KilledList.append(killer)
        self.players[killer].KillList.append(killed)
        if (active_kill):
            self.players[killer].Kill += 2
        else:
            self.players[killer].Kill += 1

    def __remove_head__(self, player_ind):
        player = self.players[player_ind]
        head_len = player.move_len
        self.map.SnakePosition[player_ind] = self.map.SnakePosition[player_ind][head_len:]
        player.total_len -= head_len
        if (player.total_len < 1):
            player.NowDead = True
        # player.head_len = 0
        player.remove_head = False
    
    def __update_time__(self):
        self.map.Time += 1
        for player in self.players:
            player.Prop['speed'] = max(0, player.Prop['speed'] - 1)
            if (player.Prop['speed'] == 0):
                player.Speed = 1
            player.Prop['strong'] = max(0, player.Prop['strong'] - 1)
            player.Prop['double'] = max(0, player.Prop['double'] - 1)

    def __get_props__(self):
        for i in range(self.player_num):
            if (self.players[i].NowDead):
                continue
            valid_poss_set = set(self.map.SnakePosition[i][:self.players[i].total_len])
            if (self.players[i].Prop['double'] > 0):
                increment = 2
            else:
                increment = 1
            for pos in valid_poss_set & self.map.SugarPosition:
                self.map.SugarPosition.remove(pos)
                self.players[i].SaveLength += increment
            for pos in valid_poss_set & self.map.PropPosition[0]:
                self.map.PropPosition[0].remove(pos)
                self.players[i].Speed += 1
                self.players[i].Prop['speed'] += 5
            for pos in valid_poss_set & self.map.PropPosition[1]:
                self.map.PropPosition[1].remove(pos)
                self.players[i].Prop['strong'] += 5
            for pos in valid_poss_set & self.map.PropPosition[2]:
                self.map.PropPosition[2].remove(pos)
                self.players[i].Prop['double'] += 5

    def __confirm_new_snakes__(self):
        # Execute death and growth
        for i in range(self.player_num):
            if (self.players[i].IsDead):
                continue
            # Death of snakes
            if (self.players[i].NowDead):
                self.__on_die__(i)
                self.players[i].IsDead = self.players[i].NowDead
                continue
            # Growth of snakes
            use_save_len = min(self.players[i].SaveLength, self.players[i].move_len)
            self.players[i].SaveLength -= use_save_len
            for _ in range(self.players[i].move_len - use_save_len):
                self.map.SnakePosition[i].pop()

    def __on_die__(self, i):
        valid_poss_set = set(self.map.SnakePosition[i][:self.players[i].total_len])
        for pos in self.map.vacant_subset(valid_poss_set):
            self.map.SugarPosition.add(pos)
        self.map.SnakePosition[i].clear()

    def __calc_scores__(self):
        for i in range(self.player_num):
            if (not self.players[i].IsDead):
                self.players[i].Score_kill = self.players[i].Kill
                self.players[i].Score_time = self.map.Time + 1
                self.players[i].Score_len = len(self.map.SnakePosition[i])
        
        scores = [0 for _ in range(self.player_num)]
        scores += 1.5 * rankdata([self.players[i].Score_kill for i in range(self.player_num)])
        scores += rankdata([self.players[i].Score_time for i in range(self.player_num)])
        scores += rankdata([self.players[i].Score_len for i in range(self.player_num)])
        scores /= 3.5

        for i in range(self.player_num):
            if (not self.players[i].IsDead):
                self.players[i].Score = scores[i]

    def __check_all_dead__(self):
        return np.all([player.IsDead for player in self.players])

    def __kill_all__(self):
        for i in range(self.player_num):
            if (not self.players[i].IsDead):
                self.players[i].IsDead = True
                self.__on_die__(i)

# %%
if __name__ ==  '__main__':
    pass
    # # Print map from gameinfo received
    # m = Map(6)
    # # m.load_game_info('game_info_sample/GameInfo_2022-04-03_11-55-55.149419.json')
    # m.load_game_info('game_info_sample/GameInfo_2022-04-03_11-55-59.513025.json')
    # m.print()

    # # Print walls and props
    # m = Map(6)
    # filelist = os.listdir('game_info_sample/')
    # for file in filelist:
    #     m.load_game_info(os.path.join('game_info_sample', file))
    #     os.system('cls')
    #     print(f'Round: {m.Time}')
    #     m.print(wall=True, sugar=False, prop=True, snake=False)
    #     sleep(1.0)


    # Inspect generation of props and sugars
    # input_dir = 'game_info_sample/'
    # input_dir = r'D:\zzx\Desktop\tmp\game_info_sample_20220404_1'
    input_dir = 'game_info_test/'
    m = Map(6)
    filelist = sorted(os.listdir(input_dir))
    pre_props = [set(), set(), set()]
    pre_sugars = set()
    diffs = []
    for file in filelist:
        m.load_game_info(os.path.join(input_dir, file))
        props = m.PropPosition
        sugars = m.SugarPosition
        # if (pre_props is not None):
        diff = [0, 0, 0, 0]
        for i in range(3):
            diff[i] = len(props[i] - pre_props[i])
        diff[3] = len(sugars - pre_sugars)
        diffs.append(diff)
        print(f'Round {m.Time}, diff = {str(diff)}, diff_sum = {str(np.sum(diffs, axis=0))}')
        pre_props = props
        pre_sugars = sugars
    diff_sum = np.sum(diffs[1:], axis=0)
    diff_sum[3] -= np.sum(np.sort(np.array(diffs)[1:, 3])[-6:])
    print(f'diff_sum = {str(diff_sum)}')

    # game = SnakeGame([AI0 for _ in range(6)])
    # game.run_till_end(savedir='game_info_test')


# %%
game = SnakeGame([AI0 for _ in range(6)])
game.run_till_end(savedir='game_info_test')

# %%
