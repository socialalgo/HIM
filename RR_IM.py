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
        path = "network_scale.csv"
        print("Reading the abstract of graph...")

        n = pd.read_csv(path, header=None).values.tolist()[0][0]

        self.edges = [[] for _ in range(n)]
        self.edges_T = [[] for _ in range(n)]
        self.degrees = [0 for _ in range(n)]
        self.nodes = [i for i in range(n)]

    def read_attribute(self, args):
        path = "deg_list.csv"  # second file of kp input
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
        path = "edge_list.csv"  # third file of kp input
        print("Reading the linklist of graph...")

        cnt = 0

        count = [0 for _ in range(self.number_of_nodes())]
        count_T = [0 for _ in range(self.number_of_nodes())]

        edge_list = pd.read_csv(path, header=None).values.tolist()

        for row in edge_list:
            u, v, w = int(row[0]), int(row[1]), float(row[2])

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

    seed_path = "ap_list.csv"
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
        result.append(u)
        for v, w in g.edges_T[u]:
            if v not in vis:
                if np.random.uniform(0, 1) <= w:
                    vis.add(v)
                    que.append(v)
    return tuple(result)


def RR_set_IC_instance_given_node(g, node):
    """返回随机一个节点的可达集合"""

    start = node
    vis = {start}
    que, head = [start], 0
    result = []
    while head < len(que):
        u = que[head]
        head = head + 1
        result.append(u)
        for v, w in g.edges_T[u]:
            if v not in vis:
                if np.random.uniform(0, 1) <= w:
                    vis.add(v)
                    que.append(v)
    return tuple(result)


def GIM_global_selection(g, ap, R):
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

        if cover_num_v == 0:  # cannot find any marginal gain, stop iteration
            print("no marginnal gain, out of iteration")
            break

        # enumerate this node's valid ap
        source_of_v = [u for u, _ in g.edges_T[v] if u in ap_selected]

        # identify ap which appear in v's vaild RR sets
        tmp = [w for rr_index in covered[v] if rr_index not in RR_set_covered for w in R[rr_index] if w in source_of_v]

        # true_aps = list(set(tmp))

        true_aps = Counter(tmp)

        if len(true_aps) == 0:
            continue

        # update influence
        cnt += 1
        current_influence += cover_num[v]

        # append result
        for ap in true_aps:
            results.append((v, ap, true_aps[ap], cnt))

        # update RR set
        for rr_index in covered[v]:
            if rr_index in RR_set_covered:
                continue
            for w in R[rr_index]:
                cover_num[w] -= 1
            RR_set_covered.add(rr_index)

        # information for debug
        if cnt <= 3:
            print(cnt)
            print("current node: " + str(v))
            print("current cover num: " + str(-cover_num_v))
            print("len of source_of_v: " + str(len(source_of_v)))
            print("len of true_aps: " + str(len(true_aps)))

    print("marginal gain is 0, num of dst node: " + str(cnt))

    return results, current_influence


def run_RR_IM(g, ap, args):
    # 得到所有的节点集合，并计算生成的rr-set数量（总节点数*每个节点的采样数 upper_ratio）
    node_set = g.nodes
    theta_0 = len(node_set) * args.sample_number
    theta_0 = int(theta_0)

    print("# of nodes: ", len(node_set))
    print("inital rr set: " + str(theta_0))
    friends_list = ''

    print("generating rr set: " + str(theta_0))

    R1 = [None] * theta_0
    cur = 0
    # generate rr set

    t_start = time.time()
    for i, node in enumerate(node_set):
        for j in range(int(args.sample_number)):
            R1[cur] = RR_set_IC_instance_given_node(g, node)
            cur += 1

            if (cur + 1) % 500000 == 0:
                print(str(cur + 1) + "/" + str(theta_0) + "...")

    t_end = time.time()
    print('complete RR set generation, using ' + str(t_end - t_start) + 's')

    friends_list, upperC = GIM_global_selection(g, ap, R1)

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
        "--sample_number",
        default=1,
        type=float
    )
    args, _ = parser.parse_known_args()

    g = Graph_S3(args)
    print("reading graph done.")
    print("total edges: " + str(g.number_of_edges()))
    print("total nodes: " + str(g.number_of_nodes()))

    sources = get_source(args)
    print("num of sources: " + str(len(sources)))

    res = run_RR_IM(g, sources, args)

    with open("id2uid.json", "r") as fr:
        id2uid = json.load(fr)

    rec_data = []
    for rec in res:
        uid = id2uid[str(rec[0])]
        frienduid = id2uid[str(rec[1])]
        number_of_shared_rr_set = rec[2]
        rec_data.append([uid,frienduid, number_of_shared_rr_set])

    df_rec = pd.DataFrame(rec_data, columns=['uid', 'frienduid', 'rr_shared'])
    df_rec.to_csv("rec_result_rr_im.csv", header=True, index=False, encoding='utf-8')