import math
import os
import random
from logging import DEBUG, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr.common
import numpy as np
import pandas
import torch
import torchvision
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
    ndarrays_to_parameters, parameters_to_ndarrays,
    MetricsAggregationFn, NDArrays, Parameters, Scalar,
)
from flwr.common.logger import log
from flwr.server import SimpleClientManager
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

import client as clt
from constants import POOL_SIZE, NUM_ROUNDS, NUM_ON_STRIKE, NUM_TO_CHOOSE, ALPHA, V_THRESHOLD, BETA, SERVER_CAPACITY
from dataset_utils import cifar10Transformation
from game_theory import game_play, time_cost_calc


class C2MAB_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients[str(0)]] # Designating client "0", this is for precisely retrieving its initial loss

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients which meet the criterion
        param_dicts = []
        sorted_cids = [] # For Stage 1
        selected_cids = [] # For Stage 2 (Final)

        ###### STAGE 1 - Evaluation of Generalization ######################################################

        # Get info of previous round
        # if server_round == 1:
        #     init_parameters = self.clients["0"].get_parameters(
        #         ins=GetParametersIns(config={}),
        #         timeout=None
        #     ).parameters
        #     init_param_ndarrays = parameters_to_ndarrays(init_parameters)
        #     init_eval_func = clt.get_evaluate_fn(
        #         torchvision.datasets.CIFAR10(
        #             root="./data", train=False, transform=cifar10Transformation()
        #         )
        #     )
        #     eval_res = init_eval_func(0, init_param_ndarrays, {})
        #     initial_loss = eval_res[0]

        L_it = [1 for _ in range(POOL_SIZE)] # The mean square batch loss of each client
        G_it = [0 for _ in range(POOL_SIZE)] # The generalization evaluation of each client

        # Get each client's parameters
        for n in range(POOL_SIZE):
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )
            param_dicts[n]["isSelected"] = False

        # Volatility
        active_cids = list(range(POOL_SIZE)) # Firstly, get a list of all client IDs
        for _ in range(NUM_ON_STRIKE):
            pop_idx = random.randint(0, len(active_cids) - 1) # The index of the client IDs to be removed
            active_cids.pop(pop_idx)

        # 1st iteration: data size only
        if server_round == 1:
            for i in range(POOL_SIZE):
                G_it[i] = param_dicts[i]["dataSize"]
                param_dicts[i]["involvement_history"] = 0
        # Common cases
        else:
            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            valid_loss_sum = 0.0
            for n in range(POOL_SIZE):
                if n in cids_in_prev_round:
                    with open("./output/mean_square_batch_loss/client_{}.txt".format(n)) as inputFile:
                        L_it[n] = eval(inputFile.readlines()[-1])
                        valid_loss_sum += L_it[n]
            
            for n in range(POOL_SIZE):
                if n not in cids_in_prev_round:
                    L_it[n] = valid_loss_sum / len(cids_in_prev_round)

            with open("./output/involvement_history.txt", mode='r') as inputFile:
                involvement_history = eval(inputFile.readline())
            for i in range(POOL_SIZE):
                param_dicts[i]["involvement_history"] = involvement_history[i]
            log(DEBUG, "Involvement history: " + str(involvement_history))

            for i in range(POOL_SIZE):
                G_it[i] = param_dicts[i]["dataSize"] * L_it[i] / ((param_dicts[i]["involvement_history"] + 1) ** ALPHA)

        sorted_cids = sorted(active_cids, key=lambda i: G_it[i], reverse=True) # Sort according to the generalization evaluation

        ###### END OF STAGE 1 ##############################################################################

        ###### STAGE 2 - Selection of Clients ##############################################################

        # C2MAB
        list_device_context_vec_c = []
        for i in range(POOL_SIZE):
            list_device_context_vec_c.append(np.array([
                [param_dicts[i]["dataSize"]],
                [1 / param_dicts[i]["splitLayer"]],
                [param_dicts[i]["computation"]],
                [param_dicts[i]["transPower"]]
            ]))

        list_context_mat_A = []
        list_context_vec_g = []

        for i in range(POOL_SIZE):

            if not os.path.exists("./output/client_context_mat_A/client_{}.npy".format(i)):
                list_context_mat_A.append(np.eye(4))
            else:
                list_context_mat_A.append(np.load("./output/client_context_mat_A/client_{}.npy".format(i)))

            if not os.path.exists("./output/client_context_vec_g/client_{}.npy".format(i)):
                list_context_vec_g.append(np.zeros((4, 1)))
            else:
                list_context_vec_g.append(np.load("./output/client_context_vec_g/client_{}.npy".format(i)))
        
        list_theta_hat = []
        for i in range(POOL_SIZE):
            list_theta_hat.append(np.matmul(
                np.linalg.inv(list_context_mat_A[i]),
                list_context_vec_g[i]
            ))

        list_V = []
        for i in range(POOL_SIZE):
            list_V.append(
                np.matmul(
                    np.transpose(list_device_context_vec_c[i]),
                    list_theta_hat[i]
                ) + 
                np.sqrt(
                    np.matmul(
                        np.matmul(
                            np.transpose(list_device_context_vec_c[i]),
                            np.linalg.inv(list_context_mat_A[i])
                        ),
                        list_device_context_vec_c[i]
                    )
                ) * BETA
            )

            param_dicts[i]["reward"] = np.matmul(
                np.transpose(list_device_context_vec_c[i]),
                list_theta_hat[i]
            )[0][0]
        
        cids_sorted_by_V = sorted(list(range(POOL_SIZE)), key=lambda i: list_V[i], reverse=True)

        for i in sorted_cids:
            if i in cids_sorted_by_V[ : int(POOL_SIZE * V_THRESHOLD)]:
                selected_cids.append(i)
                param_dicts[i]["isSelected"] = True
            if len(selected_cids) == NUM_TO_CHOOSE:
                break
        
        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(selected_cids))

        ###### END OF STAGE 2 ##############################################################################

        ###### STAGE 3 - Resource Allocation based on Game Theory ##########################################

        offload_flag_dict = game_play(selected_cids, param_dicts)
        max_time_cost, time_cost_dict = time_cost_calc(selected_cids, param_dicts, offload_flag_dict)

        log(DEBUG, "Round " + str(server_round) + " offloading plan: " + str(offload_flag_dict))

        ###### END OF STAGE 3 ##############################################################################

        # Update context

        new_list_context_mat_A = []
        new_list_context_vec_g = []

        for i in range(POOL_SIZE):

            new_list_context_mat_A.append(list_context_mat_A[i] + np.matmul(
                list_device_context_vec_c[i],
                np.transpose(list_device_context_vec_c[i])
            ))

            np.save("./output/client_context_mat_A/client_{}.npy".format(i), new_list_context_mat_A[i])

            new_list_context_vec_g.append(list_context_vec_g[i] + list_device_context_vec_c[i] * list_V[i])

            np.save("./output/client_context_vec_g/client_{}.npy".format(i), new_list_context_vec_g[i])

        return [self.clients[str(cid)] for cid in selected_cids], \
            {
                "clients_selected": selected_cids,
                "time_elapsed": max_time_cost,
                "time_constraint": time_constr
            }, \
            param_dicts


class Random_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients[str(0)]] # Designating client "0", this is for precisely retrieving its initial loss

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients randomly
        param_dicts = []
        selected_cids = []

        # Get each client's parameters
        for n in range(POOL_SIZE):
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )
            param_dicts[n]["isSelected"] = False

        # Volatility
        active_cids = list(range(POOL_SIZE)) # Firstly, get a list of all client IDs
        for _ in range(NUM_ON_STRIKE):
            pop_idx = random.randint(0, len(active_cids) - 1) # The index of the client IDs to be removed
            active_cids.pop(pop_idx)

        # Random selection
        cids_tbd = active_cids.copy()
        for _ in range(POOL_SIZE - NUM_ON_STRIKE - NUM_TO_CHOOSE):
            pop_idx = random.randint(0, len(cids_tbd) - 1)
            cids_tbd.pop(pop_idx)

        selected_cids = cids_tbd.copy()
        assert len(selected_cids) == NUM_TO_CHOOSE
        
        for i in selected_cids:
            param_dicts[i]["isSelected"] = True
        
        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(selected_cids))

        ###### Resource Allocation based on Game Theory #####################################################

        # TODO Play game or not!
        # offload_flag_dict = game_play(selected_cids, param_dicts)

        offload_flag_dict = {cid: 0 for cid in selected_cids}
        offload_candidates = random.sample(selected_cids, SERVER_CAPACITY)
        for cid in offload_candidates:
            offload_flag_dict[cid] = 1

        max_time_cost, time_cost_dict = time_cost_calc(selected_cids, param_dicts, offload_flag_dict)

        log(DEBUG, "Round " + str(server_round) + " offloading plan: " + str(offload_flag_dict))

        ###### END ##########################################################################################

        return [self.clients[str(cid)] for cid in selected_cids], \
            {
                "clients_selected": selected_cids,
                "time_elapsed": max_time_cost,
                "time_constraint": time_constr
            }, \
            param_dicts


class FedCS_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients[str(0)]] # Designating client "0", this is for precisely retrieving its initial loss

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample the best clients after one-shot exploration
        param_dicts = []
        selected_cids = []

        # Get each client's parameters
        for n in range(POOL_SIZE):
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )
            param_dicts[n]["isSelected"] = False

        # Volatility
        ##### ATTENTION ####################################################################################
        if NUM_ON_STRIKE != 0:
            raise ValueError("NUM_ON_STRIKE must be 0 for FedCS")
        ####################################################################################################
        active_cids = list(range(POOL_SIZE)) # Firstly, get a list of all client IDs
        for _ in range(NUM_ON_STRIKE):
            pop_idx = random.randint(0, len(active_cids) - 1) # The index of the client IDs to be removed
            active_cids.pop(pop_idx)

        if server_round == 1:

            selected_cids = active_cids.copy()

        else:

            with open("./output/fit_server/round_1.txt") as inputFile:
                line_dict = eval(inputFile.readline())
                cids_of_r1 = line_dict["clients_selected"]
                time_cost_of_r1 = line_dict["time_cost_all"]

            cids_sorted_by_r1 = sorted(cids_of_r1, key=lambda i: time_cost_of_r1[i], reverse=False)
            selected_cids = cids_sorted_by_r1[ : NUM_TO_CHOOSE]

        # TODO The different FedCS strategies!

        # # Volatility
        # active_cids = list(range(POOL_SIZE)) # Firstly, get a list of all client IDs
        # for _ in range(NUM_ON_STRIKE):
        #     pop_idx = random.randint(0, len(active_cids) - 1) # The index of the client IDs to be removed
        #     active_cids.pop(pop_idx)

        # selected_cids = sorted(active_cids, key=lambda i: param_dicts[i]["computation"], reverse=True)[:NUM_TO_CHOOSE]
        
        for i in selected_cids:
            param_dicts[i]["isSelected"] = True
        
        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(selected_cids))

        ###### Resource Allocation based on Game Theory #####################################################

        offload_flag_dict = {}

        if server_round == 1:
            for i in selected_cids:
                offload_flag_dict[i] = 0

        else:
            # TODO Play game or not!
            # offload_flag_dict = game_play(selected_cids, param_dicts)
            
            offload_flag_dict = {cid: 0 for cid in selected_cids}
            offload_candidates = random.sample(selected_cids, SERVER_CAPACITY)
            for cid in offload_candidates:
                offload_flag_dict[cid] = 1

        max_time_cost, time_cost_dict = time_cost_calc(selected_cids, param_dicts, offload_flag_dict)

        log(DEBUG, "Round " + str(server_round) + " offloading plan: " + str(offload_flag_dict))

        ###### END ##########################################################################################

        return [self.clients[str(cid)] for cid in selected_cids], \
            {
                "clients_selected": selected_cids,
                "time_elapsed": max_time_cost,
                "time_cost_all": time_cost_dict, # Detailed time cost for each client
                "time_constraint": time_constr
            }, \
            param_dicts


class Oort_ClientManager(SimpleClientManager):
    def sample(self, num_clients: int, server_round=0, time_constr=0):
        # For model initialization
        if num_clients == 1:
            # return [self.clients[str(random.randint(0, pool_size - 1))]]
            return [self.clients[str(0)]] # Designating client "0", this is for precisely retrieving its initial loss

        # For evaluation, use the same devices as in the fit round
        elif num_clients == -1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round),
                    mode='r'
            ) as inputFile:
                cids_in_fit = eval(inputFile.readline())["clients_selected"]
            return [self.clients[str(cid)] for cid in cids_in_fit]

        # Sample clients which meet the criterion
        param_dicts = []
        selected_cids = []

        ###### STAGE 1 - Evaluation of Utility ######################################################

        # Get info of previous round
        # if server_round == 1:
        #     init_parameters = self.clients["0"].get_parameters(
        #         ins=GetParametersIns(config={}),
        #         timeout=None
        #     ).parameters
        #     init_param_ndarrays = parameters_to_ndarrays(init_parameters)
        #     init_eval_func = clt.get_evaluate_fn(
        #         torchvision.datasets.CIFAR10(
        #             root="./data", train=False, transform=cifar10Transformation()
        #         )
        #     )
        #     eval_res = init_eval_func(0, init_param_ndarrays, {})
        #     initial_loss = eval_res[0]

        L_it = [np.NaN for _ in range(POOL_SIZE)] # The mean square batch loss of each client

        # Get each client's parameters
        for n in range(POOL_SIZE):
            param_dicts.append(
                self.clients[str(n)].get_properties(
                    flwr.common.GetPropertiesIns(config={}), 68400
                ).properties.copy()
            )
            param_dicts[n]["isSelected"] = False

        # Volatility
        active_cids = list(range(POOL_SIZE)) # Firstly, get a list of all client IDs
        for _ in range(NUM_ON_STRIKE):
            pop_idx = random.randint(0, len(active_cids) - 1) # The index of the client IDs to be removed
            active_cids.pop(pop_idx)

        # Dealing with involvement history

        # 1st iteration: no involvement history
        if server_round == 1:

            for i in range(POOL_SIZE):
                param_dicts[i]["involvement_history"] = 0
                param_dicts[i]["last_involving_round"] = -1

            log(DEBUG, "Involvement history: " + str([param_dicts[i]["involvement_history"] for i in range(POOL_SIZE)]))
            log(DEBUG, "Last involving round: " + str([param_dicts[i]["last_involving_round"] for i in range(POOL_SIZE)]))

        # Common cases
        else:

            with open("./output/fit_server/round_{}.txt".format(server_round - 1)) as inputFile:
                cids_in_prev_round = eval(inputFile.readline())["clients_selected"]

            for n in range(POOL_SIZE):
                if n in cids_in_prev_round:
                    with open("./output/mean_square_batch_loss/client_{}.txt".format(n)) as inputFile:
                        L_it[n] = eval(inputFile.readlines()[-1])
                else:
                    L_it[n] = np.NaN

            with open("./output/involvement_history.txt", mode='r') as inputFile:
                involvement_history = eval(inputFile.readline())
            for i in range(POOL_SIZE):
                param_dicts[i]["involvement_history"] = involvement_history[i]

            with open("./output/last_involving_round.txt", mode='r') as inputFile:
                last_involving_round = eval(inputFile.readline())
            for i in range(POOL_SIZE):
                param_dicts[i]["last_involving_round"] = last_involving_round[i]

            log(DEBUG, "Involvement history: " + str(involvement_history))
            log(DEBUG, "Last involving round: " + str(last_involving_round))

        # Calculating utility
        oort_epsilon = 0.1
        oort_alpha = 2

        num_unexplored_to_choose = int(NUM_TO_CHOOSE * oort_epsilon)
        num_explored_to_choose = NUM_TO_CHOOSE - num_unexplored_to_choose

        unexplored_cids = []
        explored_cids = []

        for i in range(POOL_SIZE):

            # If the client was not involved in the previous round, it is unexplored
            if np.isnan(L_it[i]):
                assert param_dicts[i]["last_involving_round"] < server_round - 1
                unexplored_cids.append(i)
                param_dicts[i]["oort_utility"] = np.NaN

            # If the client was involved in the previous round, it is explored
            else:
                assert param_dicts[i]["last_involving_round"] == server_round - 1
                explored_cids.append(i)
                param_dicts[i]["oort_utility"] = L_it[i] * param_dicts[i]["dataSize"] + math.sqrt(0.1 * math.log(server_round, 10) / param_dicts[i]["last_involving_round"])

        log(DEBUG, "Round " + str(server_round) + " unexplored cids " + str(unexplored_cids))
        log(DEBUG, "Round " + str(server_round) + " explored cids " + str(explored_cids))

        # For unexplored clients
        sorted_unexplored_cids = sorted(unexplored_cids, key=lambda i: param_dicts[i]["computation"], reverse=True)
        log(DEBUG, "Round " + str(server_round) + " sorted unexplored cids " + str(sorted_unexplored_cids))

        # For explored clients
        if server_round > 1:
            
            explored_cids_sorted_by_speed = sorted(explored_cids, key=lambda i: param_dicts[i]["computation"], reverse=True)
            log(DEBUG, "Round " + str(server_round) + " explored cids sorted by speed " + str(explored_cids_sorted_by_speed))
            
            the_deadline_T = 1 / param_dicts[explored_cids_sorted_by_speed[int(len(explored_cids_sorted_by_speed) * V_THRESHOLD)]]["computation"]
            log(DEBUG, "Round " + str(server_round) + " the deadline T " + str(the_deadline_T))
            
            for i in explored_cids:
                if 1 / param_dicts[i]["computation"] > the_deadline_T:
                    assert math.pow(the_deadline_T / (1 / param_dicts[i]["computation"]), oort_alpha) < 1
                    param_dicts[i]["oort_utility"] *= math.pow(the_deadline_T / (1 / param_dicts[i]["computation"]), oort_alpha)
            log(DEBUG, "Round " + str(server_round) + " oort utility " + str([param_dicts[i]["oort_utility"] for i in range(POOL_SIZE)]))

            sorted_explored_cids = sorted(explored_cids, key=lambda i: param_dicts[i]["oort_utility"], reverse=True)
            log(DEBUG, "Round " + str(server_round) + " sorted explored cids " + str(sorted_explored_cids))

        ###### END OF STAGE 1 ##############################################################################

        ###### STAGE 2 - Selection of Clients ##############################################################

        # 1st round: only select unexplored clients as all clients are unexplored
        if server_round == 1:
            selected_cids = sorted_unexplored_cids[ : NUM_TO_CHOOSE]
        # Common cases
        else:
            selected_cids += sorted_unexplored_cids[ : num_unexplored_to_choose]
            selected_cids += sorted_explored_cids[ : num_explored_to_choose]
            assert len(selected_cids) == NUM_TO_CHOOSE

        for i in selected_cids:
            param_dicts[i]["isSelected"] = True
            param_dicts[i]["last_involving_round"] = server_round

        with open("./output/last_involving_round.txt", mode='w') as outputFile:
            outputFile.write(str([param_dicts[i]["last_involving_round"] for i in range(POOL_SIZE)]))
        
        log(DEBUG, "Round " + str(server_round) + " selected cids " + str(selected_cids))

        ###### END OF STAGE 2 ##############################################################################

        ###### STAGE 3 - Resource Allocation based on Game Theory ##########################################

        # TODO Play game or not!
        # offload_flag_dict = game_play(selected_cids, param_dicts)
        
        offload_flag_dict = {cid: 0 for cid in selected_cids}
        offload_candidates = random.sample(selected_cids, SERVER_CAPACITY)
        for cid in offload_candidates:
            offload_flag_dict[cid] = 1

        max_time_cost, time_cost_dict = time_cost_calc(selected_cids, param_dicts, offload_flag_dict)

        log(DEBUG, "Round " + str(server_round) + " offloading plan: " + str(offload_flag_dict))

        ###### END OF STAGE 3 ##############################################################################

        return [self.clients[str(cid)] for cid in selected_cids], \
            {
                "clients_selected": selected_cids,
                "time_elapsed": max_time_cost,
                "time_constraint": time_constr
            }, \
            param_dicts


class SplitFederatedLearning(Strategy):
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"Federated Learning (accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[C2MAB_ClientManager, Random_ClientManager, FedCS_ClientManager, Oort_ClientManager] # TODO: More client managers
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit_clients config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Check records of previous round
        if server_round > 1:
            with open(
                    "./output/fit_server/round_{}.txt".format(server_round - 1),
                    mode='r'
            ) as inputFile:
                clients_of_prev_round = eval(inputFile.readline())["clients_selected"]

            for _ in range(POOL_SIZE):
                # If the client was not selected in the previous round,
                # help it complete the records
                if _ not in clients_of_prev_round:
                    with open(
                            "./output/train_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_accu/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")
                    with open(
                            "./output/val_loss/client_{}.txt".format(_),
                            mode='a'
                    ) as outputFile:
                        outputFile.write("-1" + "\n")

            # Record historic involvements
            with open("./output/involvement_history.txt", mode='r') as inputFile:
                fileLine = inputFile.readline()
                if not fileLine:
                    involvement_history = [0 for _ in range(POOL_SIZE)]
                else:
                    involvement_history = eval(fileLine)
            for _ in range(POOL_SIZE):
                if _ in clients_of_prev_round:
                    involvement_history[_] += 1
            with open("./output/involvement_history.txt", mode='w') as outputFile:
                outputFile.write(str(involvement_history))

        # Sample clients
        clients, fit_round_dict, param_dicts = client_manager.sample(
            num_clients=0, server_round=server_round, time_constr=None # num_clients and time_constr are not used
        )

        # Record information of clients
        pandas.DataFrame.from_records(param_dicts).to_excel(
            "./output/fit_clients/fit_round_{}.xlsx".format(server_round)
        )

        # Record information of server
        with open(
                "./output/fit_server/round_{}.txt".format(server_round),
                mode='w'
        ) as outputFile:
            outputFile.write(str(fit_round_dict))

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: Union[C2MAB_ClientManager, Random_ClientManager, FedCS_ClientManager, Oort_ClientManager] # TODO: More client managers
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients: use same clients as in fit
        clients = client_manager.sample(
            num_clients=-1, server_round=server_round, time_constr=None # num_clients and time_constr are not used
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit_clients results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Save the aggregated model parameters
        if server_round == NUM_ROUNDS:
            torch.save(parameters_aggregated, "./output/aggregated_model.pth")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
