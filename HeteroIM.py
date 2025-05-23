from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import collections
import string
import csv
import pandas as pd
import heapq
import sys
import os
import time
import random
import pandas
import numpy as np
import datetime
from multiprocessing.pool import ThreadPool
import math
from collections import Counter
from evaluators import evaluate

class Graph_S3:
    def __init__(self, fname):

        self.edges = []
        self.edges_T = []
        self.nodes = []
        self.degrees = []
        self.m = 0

        self.read_n(args)
        self.read_attribute(args)
        self.read_graph(args)

        print("edge list length: ", len(self.edges))
        print("node list length: ", len(self.nodes))

    def read_n(self, args):
        path = "data/network_scale.csv"
        print("Reading the abstract of graph...")

        n = pd.read_csv(path, header=None).values.tolist()[0][0]

        self.edges = [[] for _ in range(n)]
        self.edges_T = [[] for _ in range(n)]
        self.degrees = [0 for _ in range(n)]
        self.nodes = [i for i in range(n)]

    def read_attribute(self, args):
        path = "data/deg_list.csv"  # second file of kp input
        print("Reading the nodes' attribute of graph...")

        """setting data format"""
        dt = np.dtype([('v', np.int64), ('w', np.float64)])
        deg_list = pd.read_csv(path, header=None).values.tolist()

        for row in deg_list:
            uid, out_deg, in_deg = int(row[0]), int(row[1]), int(row[2])

            self.degrees[uid] = out_deg
            self.edges[uid] = np.array([(0, 0) for _ in range(out_deg)], dtype=dt)
            self.edges_T[uid] = np.array([(0, 0) for _ in range(in_deg)], dtype=dt)

    def read_graph(self, args):
        path = "data/edge_list.csv"  # third file of kp input
        print("Reading the linklist of graph...")

        count = [0 for _ in range(self.number_of_nodes())]
        count_T = [0 for _ in range(self.number_of_nodes())]

        edge_list = pd.read_csv(path, header=None).values.tolist()

        for row in edge_list:
            u, v, w = int(row[0]), int(row[1]), float(row[2])

            if u == v:
                continue

            self.edges[u][count[u]] = (v, w)
            self.edges_T[v][count_T[v]] = (u, w)
            count[u] += 1
            count_T[v] += 1
            self.m += 1

    def degree(self, u):
        return self.degrees[u]

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return self.m


def get_source(args):

    seed_path = "data/ap_list_spread_final.csv"
    data = pd.read_csv(seed_path, header=None).values.tolist()
    all_sources = [_[0] for _ in data]
    print("all sources top5: " + str(all_sources[:5]))

    del data

    return all_sources


def get_seed_candidates(g, sources):
    """返回sources的一阶邻居集合"""

    candidates = set()
    for u in sources:
        for v, _ in g.edges[u]:
            candidates.add(v)
    return candidates


def RR_set_IC_instance(g):
    """返回随机一个节点的可达集合"""

    start = random.choice(g.nodes)
    vis = {start}
    que, head = [start], 0
    result = []
    while head < len(que):
        u = que[head]
        head = head + 1
        result.append(int(u))
        for v, w in g.edges_T[u]:
            if v not in vis:
                if np.random.uniform(0, 1) <= w:
                    vis.add(v)
                    que.append(v)
    # return tuple(result)
    return result


def RR_set_IC_instance_given_node(g, node):
    """返回随机一个节点的可达集合"""

    start = node
    vis = {start}
    que, head = [start], 0
    result = []
    while head < len(que):
        u = que[head]
        head = head + 1
        result.append(int(u))
        for v, w in g.edges_T[u]:
            if v not in vis:
                if np.random.uniform(0, 1) <= w:
                    vis.add(v)
                    que.append(v)
    return result

# g:Graph, ap: test nodes, R: RR sets generated
def HeteroIM_global_selection(g, ap, R):

    # select the 1st-order neighbor of test nodes as candidates
    candidate = get_seed_candidates(g, ap)
    print("candidate neighbors size: ", len(candidate))

    # temporary arrays for RR set
    RR_set_covered = set()
    cover_num = {each: 0 for each in g.nodes}
    covered = {each: [] for each in g.nodes}
    for i in range(len(R)):
        for u in R[i]:
            covered[u].append(i)
            cover_num[u] += 1

    cover_data = [cover_num[user] for user in cover_num]
    max_cover_num = np.max(cover_data)

    # tight influence = INF
    current_influence, results = 0, []
    ap_selected = {each: 0 for each in ap}

    Q = [(-cover_num[u], u) for u in candidate]
    heapq.heapify(Q)

    cnt = 0
    while len(Q) > 0:
        cover_num_v, v = heapq.heappop(Q)  # find the best candidate 'v' from RR sets

        if -cover_num_v > cover_num[v]:  # the coverage value in the priority queue is out-of-date
            heapq.heappush(Q, (-cover_num[v], v))
            continue  # pass current round

        # enumerate this node's valid ap
        source_of_v = [u for u, _ in g.edges_T[v] if u in ap_selected]

        # identify ap which appear in v's vaild RR sets
        tmp = [w for rr_index in covered[v] if rr_index not in RR_set_covered for w in R[rr_index] if w in source_of_v]
        tmp.extend(source_of_v)

        true_aps = Counter(tmp)

        # append result
        for ap in true_aps:
            cnt += 1
            if true_aps[ap] > 1:
                results.append((v, ap, true_aps[ap], cnt))
            else:
                results.append((v, ap, cover_num_v / max_cover_num, cnt))

        # update RR set
        for rr_index in covered[v]:
            if rr_index in RR_set_covered:
                continue
            for w in R[rr_index]:
                cover_num[w] -= 1
            RR_set_covered.add(rr_index)

    print("num of recommendation: " + str(cnt))

    return results

def logcnk(n_, k_):
    if k_ >= n_:
        return 0
    k_ = k_ if k_ < n_ - k_ else n_ - k_
    res_ = 0
    for i in range(1, k_ + 1):
        res_ += math.log((n_ - k_ + i) / i)
    return res_


def run_HeteroIM(g, ap, args, eps, delta):

    sum_log = 0
    for ap_i in ap:
        sum_log += logcnk(g.degree(ap_i), args.k)
    theta_0 = int(2 * (0.5 * math.sqrt(math.log(6 / delta)) + math.sqrt(0.5 * (sum_log + math.log(6 / delta)))) ** 2)

    i_max = int(math.log2(g.number_of_nodes() / eps / eps / args.k / len(ap)))
    print("rr set initial: ", theta_0)

    # calculate the rr set generate per nodes
    node_set = g.nodes
    theta_0 = theta_0 * (2**i_max)
    sample_per_node = math.ceil(theta_0 / len(node_set))

    print("imax: ", i_max)
    print("# of nodes: ", len(node_set))
    print("generating rr set: " + str(sample_per_node * len(node_set)))

    R1 = []
    cur = 0
    print(sample_per_node, math.floor(theta_0 / len(node_set)), math.ceil(theta_0 / len(node_set)))

    t_start = time.time()
    # generate rr set
    for i, node in enumerate(node_set):
        for j in range(int(sample_per_node)):
            R1.append(RR_set_IC_instance_given_node(g, node))
            cur += 1

            if (cur + 1) % 10000 == 0:
                print(str(cur + 1) + "/" + str(sample_per_node * len(node_set)) + "...")

    t_end = time.time()
    print('complete RR set generation, using ' + str(t_end - t_start) + 's')

    friends_list = HeteroIM_global_selection(g, ap, R1)

    res = []
    for v, u, score, rank in friends_list:
        res.append([u, v, score, rank])

    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str
    )
    parser.add_argument(
        "--node_num",
        type=int
    )
    parser.add_argument(
        "--k",
        default=3,
        type=float
    )
    parser.add_argument(
        "--store_rrset",  # whether save the RR set geenrated
        default=0,
        type=int
    )
    args, _ = parser.parse_known_args()

    g = Graph_S3(args)
    print("reading graph done.")
    print("total edges: " + str(g.number_of_edges()))
    print("total nodes: " + str(g.number_of_nodes()))

    sources = get_source(args)
    print("num of sources: " + str(len(sources)))

    with open("data/id2uid.json", "r") as fr:
        id2uid = json.load(fr)

    uid2id = dict()
    for id, uid in id2uid.items():
        uid2id[uid] = id

    res = run_HeteroIM(g, sources, args, 0.1, 1.0 / g.number_of_nodes())

    Recommendation = []
    for rec in res:
        uid = id2uid[str(rec[0])]
        frienduid = id2uid[str(rec[1])]
        number_of_shared_rr_set = rec[2]
        Recommendation.append([uid,frienduid, number_of_shared_rr_set])

    evaluate(Recommendation)