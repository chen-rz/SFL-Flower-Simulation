import math
import random
from model_statistics import *
from constants import *

def game_play(selected_cids: list, param_dicts: list[dict]):

    offload_flag = dict()
    for cid in selected_cids:
        offload_flag[cid] = 0
    
    available_server_blocks = SERVER_CAPACITY

    while 1:

        if available_server_blocks <= 0:
            break

        candidate_cids = [] # Record the cids that want a change

        for cid in selected_cids:
            
            # 1/3 Calculate the cost without offloading
            sm_stat = get_split_model_statistics(MODEL_TYPE, DATASET_TYPE, model_statistics(MODEL_TYPE, DATASET_TYPE)[0])
            assert sm_stat[FLP_F_S] == 0
            assert sm_stat[FLP_B_S] == 0

            time_cost_no_offld = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_D] + sm_stat[FLP_B_D]) / param_dicts[cid]["computation"]

            # 2/3 Calculate the cost with offloading
            sm_stat = get_split_model_statistics(MODEL_TYPE, DATASET_TYPE, param_dicts[cid]["splitLayer"])
            time_device = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_D] + sm_stat[FLP_B_D]) / param_dicts[cid]["computation"]
            time_server = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_S] + sm_stat[FLP_B_S]) / SERVER_COMPUTATION

            trans_intfr = 0
            for _ in selected_cids:
                if offload_flag[_] == 1 and _ != cid:
                    trans_intfr += param_dicts[_]["transPower"] * CHANNEL_GAIN
            
            trans_rate = MAX_BANDWIDTH * math.log2(1 + param_dicts[cid]["transPower"] * CHANNEL_GAIN / (BG_NOISE_POWER + trans_intfr))
            time_trans = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * sm_stat[INT_TRANS] * 2 / trans_rate

            time_cost_offld = time_device + time_server + time_trans

            # 3/3 Decide whether to change its offload_flag
            if time_cost_no_offld < time_cost_offld and offload_flag[cid] == 1:
                candidate_cids.append(cid)
            elif time_cost_offld < time_cost_no_offld and offload_flag[cid] == 0:
                candidate_cids.append(cid)
            

        if len(candidate_cids) == 0:
            break
        else:
            lucky_cid = random.choice(candidate_cids)

            if offload_flag[lucky_cid] == 0:
                available_server_blocks -= 1
            elif offload_flag[lucky_cid] == 1:
                available_server_blocks += 1

            offload_flag[lucky_cid] = 1 - offload_flag[lucky_cid]

    return offload_flag


def time_cost_calc(selected_cids: list, param_dicts: list[dict], offload_flag: dict):
    
    time_cost_dict = dict()

    for cid in selected_cids:

        if offload_flag[cid] == 0:

            sm_stat = get_split_model_statistics(MODEL_TYPE, DATASET_TYPE, model_statistics(MODEL_TYPE, DATASET_TYPE)[0])
            time_cost_dict[cid] = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_D] + sm_stat[FLP_B_D]) / param_dicts[cid]["computation"]
        
        elif offload_flag[cid] == 1:

            sm_stat = get_split_model_statistics(MODEL_TYPE, DATASET_TYPE, param_dicts[cid]["splitLayer"])
            time_device = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_D] + sm_stat[FLP_B_D]) / param_dicts[cid]["computation"]
            time_server = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * (sm_stat[FLP_F_S] + sm_stat[FLP_B_S]) / SERVER_COMPUTATION

            trans_intfr = 0
            for _ in selected_cids:
                if offload_flag[_] == 1 and _ != cid:
                    trans_intfr += param_dicts[_]["transPower"] * CHANNEL_GAIN
            
            trans_rate = MAX_BANDWIDTH * math.log2(1 + param_dicts[cid]["transPower"] * CHANNEL_GAIN / (BG_NOISE_POWER + trans_intfr))
            time_trans = (param_dicts[cid]["dataSize"] / BATCH_SIZE) * sm_stat[INT_TRANS] * 2 / trans_rate

            time_cost_dict[cid] = time_device + time_server + time_trans
    
    max_time_cost = max(time_cost_dict.values())

    return max_time_cost, time_cost_dict