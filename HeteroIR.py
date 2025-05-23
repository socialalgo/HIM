import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from evaluators import evaluate

# The implementation of Heterogeneous influence algorithm

with open("datasets/id2uid.json", "r") as fidr:
    idmap = json.load(fidr)

def HeteroIR(inter_capacity = 4):

    print("Interaction capacity: {}".format(inter_capacity))

    src_node_dict = defaultdict(list)
    HeteroInf_dict = dict() # store the heteroinf of each node

    edge_data = pd.read_csv("datasets/edge_list.csv", header=None).values.tolist()
    src_node_list = np.unique(np.array(edge_data)[:, 0].astype(np.int32))
    dst_node_list = np.unique(np.array(edge_data)[:, 1].astype(np.int32))
    node_list = np.union1d(src_node_list, dst_node_list)
    Uj_dict = dict()
    for node in node_list:
        Uj_dict[node] = 1    # For twitter data, the Uj equals 1, as no accept behavior

    for pair in edge_data:
        src = int(pair[0])
        dst = int(pair[1])
        Pij = pair[2]
        src_node_dict[src].append(Pij*Uj_dict[dst]) # calculate the PijUj of i's neighbor

    for src in src_node_dict:
        PijUj = src_node_dict[src]
        HeteroInf = np.sum(sorted(PijUj, reverse=True)[:inter_capacity])
        HeteroInf_dict[src] = HeteroInf

    # calculate βij = PijUj(1 + heteroinf(j))
    Recommendation = []
    for pair in tqdm(edge_data):
        src = idmap[str(int(pair[0]))]
        dst = idmap[str(int(pair[1]))]
        Pij = pair[2]
        influence = HeteroInf_dict[dst] if dst in HeteroInf_dict else 1e-5
        Recommendation.append([src, dst, Pij*(1 + influence)])

    print("HeteroIR finished!")

    return Recommendation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inter_capacity",
        type=int,
        default=20
    )

    args, _ = parser.parse_known_args()

    Recommendation = HeteroIR(args.inter_capacity)

    evaluate(Recommendation)

    pass