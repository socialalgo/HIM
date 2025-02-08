import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# The implementation of Heterogeneous influence algorithm
def SpreadRec(aggr_winow = 4):

    src_node_dict = defaultdict(list)
    HeteroInf_dict = dict() # store the heteroinf of each node

    edge_data = pd.read_csv("twitter_EulerNet_new_normed.csv").values.tolist()
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
        HeteroInf = np.sum(sorted(PijUj, reverse=True)[:aggr_winow])
        HeteroInf_dict[src] = HeteroInf

    # calculate the score PijUj(1 + heteroinf(j))
    SpreadRec = []
    for pair in edge_data:
        src = int(pair[0])
        dst = int(pair[1])
        Pij = pair[2]
        influence = HeteroInf_dict[dst] if dst in HeteroInf_dict else 1e-5
        SpreadRec.append([src, dst, Pij*(1 + influence)])

    df_rec = pd.DataFrame(SpreadRec, columns=['uid', 'frienduid', 'score'])
    df_rec.to_csv("rec_result_spreadrec.csv", header=True, index=False, encoding='utf-8')

    print("finished")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggr_winow",
        type=int,
        default=3
    )

    args, _ = parser.parse_known_args()

    SpreadRec(args.aggr_winow)

    pass