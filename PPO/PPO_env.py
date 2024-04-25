import numpy as np
import random
import math
from os import path


def get_dnn_param(dnn_type=None):
    if dnn_type is None:
        print("error: environment build failed due to the 'dnn-type' description was missing")
    else:
        layer_flops = []
        exit_flops = []
        exit_probability = []
        # read dnn config from files
        if dnn_type == 'AlexNet':
            layer_flops = [0, 2e6, 2e6, 2e6, 2e6, 2e6]
            exit_flops = [0, 0.67e6, 0.67e6, 0.67e6, 0.67e6, 0.67e6]
            exit_probability = [0, 0.3, 0.5, 0.7, 0.75, 1.0]
            inter_data_size = [12, 64, 128, 64, 64, 0.3]  # xxx KB
        elif dnn_type == 'VGG16':
            layer_flops = [0, 2172160, 37945344, 18833408, 37847040,
                           18989056, 37797888, 37797888, 18833408, 37773312,
                           37773312, 21247488, 21247488, 24336640]
            exit_flops = [0, 1.5e7, 1.5e7, 1.5e7, 1.5e7,
                          1.5e7, 1.5e7, 1.5e7, 1.5e7, 1.5e7,
                          1.5e7, 1.5e7, 1.5e7, 1.5e7]
            exit_probability = [0, 0.137, 0.311, 0.200, 0.431,
                                0.585, 0.723, 0.815, 0.862, 0.901,
                                0.934, 0.954, 0.957, 1.000]
            inter_data_size = [12, 256, 64, 128, 32,
                               64, 64, 16, 32, 32,
                               18, 18, 18, 0.35]
            compress_dsize = [6.4, 63.4, 43.8, 57.0, 22.7,
                              27.2, 27.6, 9.7, 10.3, 8.5,
                              5.8, 5.6, 4.0, 0.3]
        elif dnn_type == 'Inception v3':
            layer_flops = [0, 1213696, 8290560, 16798208, 4213248,
                           93562368, 147474432, 159860736, 164247552,
                           173005248, 156986368, 204611968, 204611968,
                           259176192, 120527360, 126213504, 151677056]
            exit_flops = [0, 1.28e7, 1.13e7, 1.96e7, 2.45e7,
                          4.32e7, 5.76e7, 6.48e7, 6.48e7,
                          3.63e7, 3.63e7, 3.63e7, 3.63e7,
                          3.63e7, 1.25e7, 2.0e7, 1.5e7]
            exit_probability = [0, 0.155, 0.176, 0.240, 0.324,
                                0.370, 0.408, 0.568, 0.504,
                                0.393, 0.565, 0.633, 0.780,
                                0.920, 0.931, 0.948, 1.000]
            inter_data_size = [12, 128, 113, 196, 245,
                               432, 576, 648, 648,
                               363, 363, 363, 363,
                               363, 125, 200, 0.3]
            compress_dsize = []
        elif dnn_type == 'ResNet34':
            layer_flops = [0, 2516224, 4734976, 4734976, 4734976, 3674112,
                           4726784, 4726784, 4726784, 3643392, 4722688,
                           4722688, 4722688, 4722688, 4722688, 14725120,
                           18882560, 18821376]
            exit_flops = [0, 8e5, 8e5, 8e5, 8e5,
                          8e5, 8e5, 8e5, 8e5, 8e5,
                          8e5, 8e5, 8e5, 8e5, 8e5,
                          8e5, 8e5, 8e5]
            exit_probability = [0, 0.293, 0.519, 0.668, 0.684,
                                0.849, 0.902, 0.937, 0.943, 0.949,
                                0.949, 0.950, 0.951, 0.952, 0.951,
                                0.951, 0.953, 1.0]
            inter_data_size = [12, 16, 16, 16, 16,
                               8, 8, 8, 8, 4,
                               4, 4, 4, 4, 4,
                               8, 8, 0]
            compress_dsize = []
        elif dnn_type == 'SqueezeNet1.0':
            layer_flops = [0, 2617088, 140352, 2261504, 175168,
                            2261504, 612480, 8045568, 279680,
                            1598464, 411328, 3429888, 493248,
                            3503616, 821504, 5851136, 492544,
                            2662400, 15274]
            exit_flops = [0, 7e5, 7e5, 7e5, 5e5,
                          5e5, 5e5, 7e5, 5e5, 
                          5e5, 5e5, 7e5, 5e5, 
                          7e5, 5e5, 7e5, 5e5,
                          7e5, 5e5]
            exit_probability = [0, 0.196, 0.246, 0.674, 0.561,
                                0.718, 0.700, 0.762, 0.781,
                                0.824, 0.830, 0.854, 0.842,
                                0.851, 0.849, 0.845, 0.848,
                                0.852, 1.000]
            inter_data_size = [12, 74, 12, 98, 12, 
                               98, 24, 36, 5, 
                               36, 7, 54, 7, 
                               54, 9, 32, 4, 
                               32, 0.35]
            compress_dsize = [6.0, 29.5, 7.6, 23.5, 7.0, 
                              26.7, 10.7, 9.0, 1.9,
                              7.3, 2.4, 5.2, 1.8, 
                              4.3, 1.8, 2.6, 0.8, 
                              2.7, 0.3]
        return [layer_flops, exit_flops, exit_probability, inter_data_size]


def get_device_param(device_type=None):
    if device_type is None:
        print("error: environment build failed due to the 'device-type' description was missing")
    else:
        device_capacity = []
        for device_name in device_type:
            if device_name == 'Jetson Nano':
                device_capacity.append(1e9) # not true value, for testing only
            elif device_name == 'Raspberry':
                device_capacity.append(5.9e8) # not true value, for testing only
        return device_capacity


class MultiUserEnv(object):

    def __init__(self, device_type=None, dnn_type=None, edge_capacity=None,
                 bandwidth=None, delay=None):
        # 环境属性
        self.device_param = get_device_param(device_type)
        self.dnn_param = get_dnn_param(dnn_type)
        self.edge_capacity = edge_capacity
        # self.edge_resources = edge_resources
        self.num_device = len(device_type)
        self.num_dnn_layers = len(self.dnn_param[0]) - 1
        self.bandwidth = bandwidth  #/self.num_device
        self.delay = delay
        self.worst_state = self._get_worst_state()
        # 动作空间&状态空间
        self.action_range = [1, self.num_dnn_layers]
        self.action_space = np.zeros((2, self.num_device))
        self.observation_space = np.zeros((6, self.num_device))
        # 随机化种子
        self.seed()

    # upper bound of average latency
    def _get_worst_state(self):
        # slice_point = 0
        binary_offload = []
        a1 = [[0.33, 0.33, 0.33], [0, 0, 0]]
        a2 = [[0.0, 0.0, 0.0], [self.num_dnn_layers, self.num_dnn_layers, self.num_dnn_layers]]
        u1 = np.vstack((a1[0], a1[1]))
        u2 = np.vstack((a2[0], a2[1]))
        for u in [u1, u2]:
            avg_latency = sum(self._get_cost(u))/self.num_device
            binary_offload.append(avg_latency)
        worst_latency = min(binary_offload)
        return worst_latency

        # partition1_flops = sum(self.dnn_param[0][i] for i in range(slice_point + 1))
        # partition2_flops = sum(self.dnn_param[0][i] for i in range(slice_point + 1, self.num_dnn_layers + 1))
        # local_computing = [partition1_flops / i for i in self.device_param]
        # transmission = self.dnn_param[3][slice_point] / self.bandwidth + self.delay
        # edge_computing = partition2_flops / (self.edge_capacity / self.num_device)
        # avg_latency = sum(local_computing) / self.num_device + transmission + edge_computing
        # binary_offload.append(avg_latency)

    def _get_exit_prob(self, exit_selection=None, action=None):
        if exit_selection is None:
            exit_selection = [0, ]
            # ASCII to Binary
            action = round(action)
            action = action.astype(np.int32)
            bin_str = bin(action).replace('0b', '')
            for i in range(self.num_dnn_layers-1):
                if (1 << i) & action:
                    exit_selection.append(1)
                else:
                    exit_selection.append(0)
            exit_selection.append(1)
        # get exit probability
        exit_prob_0 = list(map(lambda x, y: x * y, exit_selection, self.dnn_param[2][:]))
        exit_prob_1 = [p for p in exit_prob_0]
        for i in range(1, len(exit_prob_1)):
            exit_prob_1[i] = max(exit_prob_0[i - 1], exit_prob_1[i - 1])
        return exit_prob_1, exit_selection
    
    def _get_optimal_exits(self, dnn_slc):
        if dnn_slc < self.num_dnn_layers-1:
            if dnn_slc == 0:
                exits_part1 = [0]
            elif dnn_slc == 1:
                exits_part1 = [0, 1]
            else:
                dnn_param_part1 = [self.dnn_param[i][0:dnn_slc + 1] for i in range(3)]
                min_flops_1, exits_part1 = select_exit(dnn_param_part1)
            dnn_param_part2 = [self.dnn_param[i][dnn_slc + 1:] for i in range(3)]
            dnn_param_part2[2] = [(p - self.dnn_param[2][dnn_slc]) / (1 - self.dnn_param[2][dnn_slc])
                                  for p in dnn_param_part2[2]]
            for i in range(3):
                dnn_param_part2[i].insert(0, 0)
            min_flops_2, exits_part2 = select_exit(dnn_param_part2)
        else:
            dnn_param_part1 = [self.dnn_param[i][0:dnn_slc + 1] for i in range(3)]
            min_flops_1, exits_part1 = select_exit(dnn_param_part1)
            if dnn_slc == self.num_dnn_layers:
                exits_part2 = [0]
            elif dnn_slc == self.num_dnn_layers-1:
                exits_part2 = [0, 1]
        # print(dnn_param_part1)
        # print(dnn_param_part2)
        return exits_part1 + exits_part2[1:]
    
    def _get_cost(self, action, is_step=True):
        latency = []
        latency_norm = []
        local_latency = []
        trans_latency = []
        edge_latency = []
        # print(action)
        for col in range(action.shape[1]):
            t_local = 0
            t_edge = 0
            rs_alloc = action[0, col]
            dnn_slc = round(action[1, col])
            dnn_slc = dnn_slc.astype(np.int32)
            exit_slt_bin = self._get_optimal_exits(dnn_slc)
            exit_prob, exit_slt_bin = self._get_exit_prob(exit_slt_bin)
            miu_val = [1 - p for p in exit_prob]
            miu_val.append(0)
            # local computing latency
            for k in range(dnn_slc+1):
                t_local += miu_val[k] * (self.dnn_param[0][k] + self.dnn_param[1][k] * exit_slt_bin[k])
            t_local /= self.device_param[col]
            # transmission latency
            t_trans = miu_val[dnn_slc+1] * (self.dnn_param[3][dnn_slc] / self.bandwidth + self.delay)
            # edge computing latency
            for k in range(dnn_slc+1, self.num_dnn_layers+1):
                t_edge += miu_val[k] * (self.dnn_param[0][k] + self.dnn_param[1][k] * exit_slt_bin[k])
            if rs_alloc <= 0.01:
                rs_alloc = 0.01
            t_edge /= (rs_alloc * self.edge_capacity)
            # print('device_trans_edge: {:4f}ms, {:4f}ms, {:4f}ms'.format(t_local, t_trans, t_edge))
            T = (t_local + t_trans + t_edge)*1e3
            latency.append(T)  # 毫秒 ms
            latency_norm.append(T/31.148)
            local_latency.append(t_local*1e3/T)
            trans_latency.append(t_trans*1e3/T)
            edge_latency.append(t_edge*1e3/T)
        if is_step:
            return latency
        else:
            return latency_norm, local_latency, trans_latency, edge_latency

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # u is action, size=(3, num_device)
    def step(self, u, is_train=False):
        # previous latency
        pre_u = np.vstack((self.state[0], self.state[1]))
        pre_latency = sum(self._get_cost(pre_u))/self.num_device
        # ----------v1~v5非线性变换--------
        # u = np.abs(u)
        # ----------v6线性变换-----------
        u = (u + 1) / 2
        rs_alloc = u[0]
        dnn_slc = u[1] * self.action_range[1]
        dnn_slc = np.clip(dnn_slc, 0, self.action_range[1])
        dnn_slc = dnn_slc.astype(np.int32)
        for k in range(3):
            if dnn_slc[k] == self.action_range[1]:
                rs_alloc[k] = 0
        if np.sum(rs_alloc) > 0:
            rs_alloc = np.clip(rs_alloc/np.sum(rs_alloc), 0, 1)
        # rs_alloc = np.clip(rs_alloc, 0, 1)
        # dnn_slc = np.clip(dnn_slc * self.action_range[1], 0, self.num_dnn_layers)
        self.last_u = np.vstack((rs_alloc, dnn_slc))
        # the latency here is a list of each device
        latency = self._get_cost(self.last_u)
        goal = sum(latency)/self.num_device
        done = False
        # -----------v3------------
        # if goal < self.worst_state:
        #     costs = -math.exp(self.worst_state - goal)
        # else:
        #     costs = goal
        #------------v4------------
        # costs = 0
        # if goal < pre_latency:
        #     costs -= 5
        # else:
        #     costs += 5
        #----------- v5,v6------------
        costs = 0
        if goal < pre_latency:
            costs -= 1.0
        elif goal > pre_latency:
            costs += 1.5
        else:
            pass

        new_dnn_slc = dnn_slc
        new_rs_alloc = rs_alloc
        bw = np.full((1, self.num_device), self.bandwidth)
        dl = np.full((1, self.num_device), self.delay)
        dc = self.device_param

        self.state = np.vstack((new_rs_alloc, new_dnn_slc,
                                bw, dl, dc))
        # 返回下一时刻的观测值，回报，是否终止,调试项]
        if is_train:
            return self._get_obs(), -costs, goal, done, [rs_alloc, dnn_slc, bw, dl]
        else:
            exit_slt_bin = []
            for k in range(self.num_device):
                tmp = self._get_optimal_exits(dnn_slc[k])
                exit_slt_bin.append(tmp)
            return self._get_obs(), -costs, goal, done, [rs_alloc, dnn_slc, exit_slt_bin]

    def reset(self, network_init_=None, randomly_init_=True):
        s_width = self.num_device
        rs_alloc = np.full((1, s_width), 0.33)
        dnn_slc = np.full((1, s_width), 0)
        # rs_alloc = np.random.rand(1, s_width)
        # rs_alloc = np.clip(rs_alloc/np.sum(rs_alloc), 0, 1)
        # dnn_slc = np.array(np.random.randint(self.num_dnn_layers, size=(1, s_width)), dtype=np.int32)
        # exit_slt = self._get_optimal_exits(dnn_slc)
        bw_list = [i for i in range(1, 6)]
        dl_list = [i for i in range(1, 11)]
        if randomly_init_:
            # bw = np.random.random(1) * 20 * 1024
            # dl = np.random.random(1) / 5
            bw = np.array([random.choice(bw_list)]) * 1024
            dl = np.array([random.choice((dl_list))]) / 1000
            self.bandwidth = bw[0]
            self.delay = dl[0]
        else:
            bw = np.array([network_init_[0]]) # input bandwidth
            dl = np.array([network_init_[1]]) # input delay
            self.bandwidth = bw[0]
            self.delay = dl[0]
        # resize
        bw = np.full((1, s_width), bw)
        dl = np.full((1, s_width), dl)
        dc = self.device_param
        # bw = np.full((1, self.num_device), self.bandwidth)
        # dl = np.full((1, self.num_device), self.delay)

        # randomly reset state
        self.state = np.vstack((rs_alloc, dnn_slc,
                                bw, dl, dc))
        self.worst_state = self._get_worst_state()
        self.last_u = None
        return self._get_obs()

    # get observation
    def _get_obs(self):
        u = np.vstack((self.state[0], self.state[1]))
        latency, l_local, l_trans, l_edge = self._get_cost(u, is_step=False)
        obs = np.vstack((latency, l_local, l_trans, l_edge,
                         self.state[2]/20480, self.state[3]*1000/105))  # size=(6, NUM_DEVICE)
        return obs.reshape(-1)

    def render(self):
        return self.worst_state

