import argparse
import os
import shutil
from pathlib import Path

import flwr as fl
import torchvision

import client as clt
from dataset_utils import cifar10Transformation
from constants import POOL_SIZE, NUM_ROUNDS
from strategy import C2MAB_ClientManager, FedCS_ClientManager, Random_ClientManager, SplitFederatedLearning

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_client_cpus", type=int, default=6)
parser.add_argument("--num_client_gpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="N/A")

# Start simulation (a _default server_ will be created)
if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    fed_dir = "./data/cifar-10-batches-py/federated/"
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cifar10Transformation()
    )

    # clear previous records
    path_to_init = ["fit_clients", "fit_server",
                    "train_loss", "val_accu", "val_loss",
                    "mean_square_batch_loss", "client_context_mat_A", "client_context_vec_g"]
    for _ in path_to_init:
        if Path("output/" + _ + "/").exists():
            shutil.rmtree("output/" + _ + "/")
        os.mkdir("output/" + _ + "/")

    with open("./output/involvement_history.txt", mode='w') as outputFile:
        outputFile.write("")
    #############################################################

    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus
    }

    parameter_dict_list = []
    for _ in range(POOL_SIZE):
        parameter_dict_list.append(dict())
    with open("./parameters/dataSize.txt") as inputFile:
        for _ in range(POOL_SIZE):
            parameter_dict_list[_]["dataSize"] = eval(inputFile.readline())
    with open("./parameters/computation.txt") as inputFile:
        for _ in range(POOL_SIZE):
            parameter_dict_list[_]["computation"] = eval(inputFile.readline())
    with open("./parameters/transPower.txt") as inputFile:
        for _ in range(POOL_SIZE):
            parameter_dict_list[_]["transPower"] = eval(inputFile.readline())
    with open("./parameters/channelGain.txt") as inputFile:
        for _ in range(POOL_SIZE):
            parameter_dict_list[_]["channelGain"] = eval(inputFile.readline())


    def client_fn(cid: str):
        # create a single client instance
        return clt.FlowerClient(cid, fed_dir, parameter_dict_list[int(cid)])


    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": True,
        "log_to_driver": True
    }

    # Configure the strategy
    strategy = SplitFederatedLearning(
        on_fit_config_fn=clt.fit_config,
        # centralised evaluation of global model
        evaluate_fn=clt.get_evaluate_fn(testset),
    )

    # TODO More strategies can be added here
    # Configure the client manager
    if args.mode == "C2MAB":
        client_manager = C2MAB_ClientManager()
    elif args.mode == "Random":
        client_manager = Random_ClientManager()
    elif args.mode == "FedCS":
        client_manager = FedCS_ClientManager()
    else:
        raise ValueError("Invalid mode")

    # start simulation
    simulation = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=POOL_SIZE,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=client_manager,
        ray_init_args=ray_init_args,
    )

    print(simulation)

    with open("./output/losses_centralized.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.losses_centralized))
    with open("./output/losses_distributed.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.losses_distributed))
    with open("./output/metrics_centralized.txt", mode='w') as outputFile:
        outputFile.write(str(simulation.metrics_centralized))

    ###############################################################################
    # Check records of the last round
    with open(
            "./output/fit_server/round_{}.txt".format(NUM_ROUNDS),
            mode='r'
    ) as last_inputFile:
        clients_of_last_round = eval(last_inputFile.readline())["clients_selected"]

    for _ in range(POOL_SIZE):
        # If the client was not selected in the last round,
        # help it complete the records
        if _ not in clients_of_last_round:
            with open(
                    "./output/train_loss/client_{}.txt".format(_),
                    mode='a'
            ) as last_outputFile:
                last_outputFile.write("-1" + "\n")
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
    
    # Record involvements of last round
    with open("./output/involvement_history.txt", mode='r') as inputFile:
        fileLine = inputFile.readline()
        assert fileLine
        involvement_history = eval(fileLine)
    for _ in range(POOL_SIZE):
        if _ in clients_of_last_round:
            involvement_history[_] += 1
    with open("./output/involvement_history.txt", mode='w') as outputFile:
        outputFile.write(str(involvement_history))
    ###############################################################################
