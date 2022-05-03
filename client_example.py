#  %%
import threading
import grpc
import sys
from pathlib import Path
import numpy as np
import argparse
import time
import json

from utils import load_json

_current_root = str(Path(__file__).resolve().parents[1])
sys.path.append(_current_root)
sys.path.append('.')
print(_current_root)
import contest.snake_pb2 as dealer_pb2
import contest.snake_pb2_grpc as rpc
from lib.simple_logger import simple_logger
from datetime import datetime

import sys
sys.path.append(r'D:\zzx\Programming\vsCode\JiukunSnake')

from AI import AI_greedy_0

# %%
param_savepath_format = 'D:/zzx/Programming/vsCode/JiukunSnake/AI_greedy_0_params/params_{}.json'
params = load_json(param_savepath_format.format(8))

# %%
def AI_client(Num_,GameInfo_, tp_number):
    # time_now = str(datetime.now()).replace(' ', '_').replace(':', '-')
    # with open(f'D:/zzx/Desktop/tmp/GameInfo_{time_now}.json', 'w') as f:
    round = GameInfo_["gameinfo"]["Map"]['Time']
    with open(f'D:/zzx/Desktop/tmp/GameInfo_{tp_number}_{str(round).zfill(3)}.json', 'w') as f:
        json.dump(GameInfo_, f, indent=4)

    res = AI_greedy_0(Num_, GameInfo_, params=params, debug=False)
    player_tmp = GameInfo_["gameinfo"]["Player"][Num_]
    print('IsDead:', player_tmp["IsDead"])
    print('Score_len:', player_tmp["Score_len"])
    print('Speed:', player_tmp["Speed"])
    print('Decision =', res)
    return res

class Client(object):
    def __init__(self, username: str, key: str, logger, address='139.196.39.76', port=7777):
        self.username = username
        self.key = key
        self.address = address
        self.port = port
        # create a gRPC channel + stub
        channel = grpc.insecure_channel(self.address + ':' + str(self.port))
        self.conn = rpc.SnakeStub(channel)

        self._lock = threading.Lock()
        self._decision_so_far = []  # history of the decision info from the server
        self._is_started = True     # 控制背景心跳
        self._new_response = []     # response list from the server
        self._new_request = []      # request list waiting to send to the server

        self.init_score = 0

        self.logger = logger
        self.step = -1

        if self.logger is None:
            self.logger = simple_logger()

        self.stoped = False
        self.round = 0

        self.cond = threading.Condition()
        self.heart_beat_interval = 0.1

        self.logger.info('self.key is inited to ' + self.key)
        self.login(self.username, self.key)  # 这里是阻塞的，不登录无法进行游戏

        self._updater = threading.Thread(target=self.run)  # 维持heartbeat

        self._updater.setDaemon(True)
        self._updater.start()

    def __del__(self):
        self._is_started = False

    def login(self, user_id, user_pin):
        '''
        登录模块
        '''
        while True:
            # try:
                request = dealer_pb2.LoginRequest()
                request.user_id = user_id
                request.user_pin = user_pin
                self.logger.info('waiting for connect')
                response = self.conn.login(request)

                if response:
                    if response.success:
                        self.init_score = response.init_score
                        self.logger.info('login success, init score:%d' % self.init_score)
                        return
                    else:
                        self.logger.info('login failed.' + response.reason)
                        time.sleep(3)

            # except grpc.RpcError as error:
            #     print(error)
            #     self.logger.info('login failed. will retry one second later')
            #     time.sleep(1)

    def client_reset(self, u: str, logger):
        self.username = u
        # create a gRPC channel + stub
        channel = grpc.insecure_channel(self.address + ':' + str(self.port))
        self.conn = rpc.SnakeStub(channel)

        self._decision_so_far = []  # history of the decision info from the server
        self._new_response = []     # response list from the server
        self._new_request = []      # request list waiting to send to the server

        self.init_score = 0

        self.logger = logger
        self.step = -1
        if self.logger is None:
            self.logger = simple_logger()

        self.stoped = False
        self.round = 0

    def chat_with_server(self):
        '''
        通信相关
        '''
        while True:
            self.cond.acquire()
            while True:
                while len(self._new_request) != 0:
                    # yield a resquest from the request list to the server
                    msg = self._new_request.pop(0)
                    yield msg
                self.cond.wait()
            self.cond.release()

    def add_request(self, msg):
        self.cond.acquire()
        self._new_request.append(msg)
        self.cond.notify()
        self.cond.release()

    def run(self):
        """
        维持心跳，定期监听需要获得的消息
        """
        while self._is_started:
            # heartbeat
            msg = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                           msg_type=dealer_pb2.ActionRequest.HeartBeat)
            self.add_request(msg)

            time.sleep(self.heart_beat_interval)
            if self.stoped:
                self.client_reset(self.username, self.logger)
        return

    def start(self):
        '''
        处理从server发回的消息
        '''
        responses = self.conn.GameStream(self.chat_with_server())
        for res in responses:
            self._new_response.append(res)

            if res.msg_type == dealer_pb2.ActionResponse.GameDecision:
                # server asking for a decision from the client
                self.logger.info('request decision')

                round_output = json.loads(res.game_info)
                self.logger.info(f'user_pos: {res.user_pos}; tp_number: {res.tp_number}; total_score: {res.total_score}; total_game_number: {res.game_number}; table_round: {res.table_round}')
                ActTemp = AI_client(res.user_pos, round_output, res.tp_number)

                request = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                                   msg_type=dealer_pb2.ActionRequest.GameDecision,
                                                   game_info=ActTemp)

                self.add_request(request)
            elif res.msg_type == dealer_pb2.ActionResponse.RoundEnd:
                self.logger.info(f'{res.tp_number} game is over.')
                self.logger.info(f'user_pos: {res.user_pos}; tp_number: {res.tp_number}; total_score: {res.total_score}; total_game_number: {res.game_number}; table_round: {res.table_round}')

            elif res.msg_type == dealer_pb2.ActionResponse.StateUpdate:
                round_output = json.loads(res.game_info)
                self.step += 1

                self.logger.info('killCount is: %d' % round_output['gameinfo']['Player'][res.user_pos]['Kill'])
                self.logger.info('starCount is: %d' % round_output['gameinfo']['Player'][res.user_pos]['SaveLength'])

            elif res.msg_type == dealer_pb2.ActionResponse.StateReady:  # 询问是否准备好
                self.logger.info('request ready')
                request = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                                   msg_type=dealer_pb2.ActionRequest.StateReady)

                self.add_request(request)

            elif res.msg_type == dealer_pb2.ActionResponse.GameEnd:
                self.logger.info('game end')
                self._is_started = False
                self.stoped = True
                return



#**********************************NOTICE**************************************
# You should make sure that your username and key is right,
# You should never keep more than one connection to the server at the same time.
#******************************************************************************


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument('--username', type=str)
    parser.add_argument('--key', type=str)
    parser.add_argument('--address', type=str, default='139.196.39.76')
    parser.add_argument('--port', type=int, default=7777)
    args = parser.parse_args()

    logger = simple_logger()

    c = Client(args.username, args.key, logger, args.address, args.port)
    c.start()
    exit()

