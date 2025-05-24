import os.path

import numpy as np
import pandas as pd
import json
import itertools
from collections import defaultdict
from tqdm import tqdm

def spread_num_process():

    spread_Data = pd.read_csv("datasets/twitter_spread.csv")
    spread = np.array(spread_Data)

    spread_num = defaultdict(int)
    for pair in spread:
        src = pair[0]
        dst = pair[1]
        if src != dst:
            spread_num[src] = spread_num[src] + 1

    return spread_num

def Theory_maximum_rec():

    Kmax = 3

    spread_Data = pd.read_csv("datasets/twitter_spread.csv")

    col0 = spread_Data.columns[0]
    col1 = spread_Data.columns[1]

    src_list = spread_Data[col0]
    dst_list = spread_Data[col1]
    spread = []
    for (src, dst) in zip(src_list, dst_list):
        spread.append([src, dst])

    spread_dict = defaultdict(list) # whom be spread by invitor
    spread_num = defaultdict(int)   # spread number of each invitor

    upper_lim = defaultdict(dict)  # ISpread@K per user
    upper_user = defaultdict(dict)  # whom to recommend to get ISpread@K per user
    for pair in spread:
        src = pair[0]   # invitor
        dst = pair[1]   # invitee
        if src != dst:
            spread_dict[src].append(dst)
            spread_num[src] = spread_num[src] + 1

    for user in spread_dict:
        upper_lim[user] = [-1 for _ in range(Kmax)]
        upper_user[user] = [[] for _ in range(Kmax)]

    to_be_rec = list(spread_dict.keys())    # test node
    for node in tqdm(to_be_rec):
        for K in range(1, Kmax + 1):  # Spread@K

            neighbor = spread_dict[node]  # whom be spread
            if len(neighbor) < K:         # If no more than K being spread
                upper_lim[node][K - 1] = upper_lim[node][K - 2]
                upper_user[node][K - 1] = upper_user[node][K - 2]
                continue

            # Sample K user from neighbor by permutation to get the best results
            full_sample_list = list(itertools.combinations(neighbor, K))

            if len(full_sample_list) > 0:
                best_cover = 0
                best_rec = []
                for lst in full_sample_list:                 # iterate all the possible recommendation
                    cur = []
                    for _right in lst:                       # iterate each node in sample result
                        if _right in spread_dict:            # if invitee has secondary spread
                            cur.extend(spread_dict[_right])  # record the secondary spread

                    cur.extend(lst)
                    unique_cur = np.unique(cur)  # reduplication

                    # Update the spread coverage
                    if len(unique_cur) > best_cover:
                        best_cover = max(best_cover, len(unique_cur)) if node not in unique_cur else max(best_cover,
                                                                                                     len(unique_cur) - 1)
                        best_rec = lst

            upper_lim[node][K - 1] = best_cover
            upper_user[node][K - 1] = best_rec

    return upper_user

def ISpread_K_calculated():

    Kmax = 3
    ISpread_K = []

    # test nodes
    ap_file = "datasets/ap_list_spread_final.csv"
    test_node = [_ for _ in np.array(pd.read_csv(ap_file, header=None)).reshape(-1, )]

    spread_Data = pd.read_csv("data/twitter_spread.csv")

    upper_limit_user = Theory_maximum_rec()
    print("Theory recommend list generated!")

    limit_dict = spread_num_process()
    print("Effective spread capability generated!")

    col0 = spread_Data.columns[0]
    col1 = spread_Data.columns[1]

    src_list = spread_Data[col0]
    dst_list = spread_Data[col1]
    spread = []
    for (src, dst) in zip(src_list, dst_list):
        spread.append([src, dst])

    spread_dict = defaultdict(list)
    spread_num = defaultdict(int)
    for pair in spread:
        src = pair[0]
        dst = pair[1]
        if src != dst:
            spread_dict[src].append(dst)
            spread_num[src] = spread_num[src] + 1

    to_be_rec = test_node
    for K in range(1, Kmax + 1):
        rec_list = []
        for _, node in enumerate(to_be_rec):
            if node not in upper_limit_user:
                continue
            theory_upper = upper_limit_user[node]
            if limit_dict[node] >= K:
                rec_list.extend(theory_upper[K - 1])
            else:
                rec_list.extend(theory_upper[limit_dict[node] - 1])

        rec_list = np.unique(rec_list)  # deduplicate
        second_coverage = []
        second_coverage.extend(rec_list)    # extend first-order spread from inviter to invitees
        for rec in rec_list:
            second_coverage.extend(spread_dict[rec])
        print("ISpread@{} = {}".format(K, len(np.unique(second_coverage))))

        ISpread_K.append(len(np.unique(second_coverage)))

    df = pd.DataFrame(ISpread_K)
    df.to_csv("datasets/ISpread@K.csv", index=False, header=False)

def evaluate(Recommendation):

    print("Evaluation Begins")

    # Get idmap
    with open("datasets/id2uid.json", "r") as fidr:
        idmap = json.load(fidr)

    # network & probability predicted by EulerNet
    edge_list = np.array(pd.read_csv("data/edge_list.csv", header=None))
    frienddict = defaultdict(list)
    for src, dst, p in edge_list:
        frienddict[idmap[str(int(src))]].append(idmap[str(int(dst))])
    friendnum = defaultdict(int)
    for uid in frienddict:
        friendnum[uid] = len(frienddict[uid])

    spread_Data = pd.read_csv("datasets/twitter_spread.csv")

    # test node file
    ap_file = "datasets/ap_list_spread_final.csv"
    test_left_node = [idmap[str(_)] for _ in np.array(pd.read_csv(ap_file, header=None)).reshape(-1, )]
    test_node_dict = dict()
    for node in test_left_node:
        test_node_dict[node] = 1

    # verify the friend number with at least 3
    ds = [len(frienddict[src]) for src in test_left_node]
    assert min(ds) == 3

    spread = np.array(spread_Data)
    spread_dict = defaultdict(list) # whom be spread by the invitor
    spread_num = defaultdict(int)   # the quantity of the invitees spread by the invitor
    for pair in spread:
        src = pair[0]
        dst = pair[1]
        if src != dst:
            spread_dict[src].append(dst)  # whom be spread
            spread_num[src] = spread_num[src] + 1  # effective spread of the inviter

    test_data = Recommendation

    test_user_dict = defaultdict(list)

    for src, dst, v in tqdm(test_data):
        src = int(src)
        dst = int(dst)
        if src in test_node_dict:
            test_user_dict[src].append([dst, v])    # get the recommendation results

    # Recall@K, NDCG@k
    K = 3
    Recall_K = [[] for _ in range(K)]
    DCG_K = [[0 for __ in range(len(test_user_dict))] for _ in range(K)]
    NDCG_K = [[] for _ in range(K)]

    Spread_K_valid = dict()   # recommend the valid recommendation in TopK
    for user in test_user_dict:
        Spread_K_valid[user] = [[], [], []]

    Recall_K_deg = defaultdict(list)

    # predefine the idcg@K of each user
    idcg_data = defaultdict(list)   # n*3, [[u1_idcg@1, u1_idcg@2, u1_idcg@3], ...[un_idcg@1, un_idcg@2, un_idcg@3]]
    idcg = [1/np.log2(2), 1/np.log2(2) + 1/np.log2(3), 1/np.log2(2) + 1/np.log2(3) + 1/np.log2(4)]  # init idcg
    for user in test_user_dict:
        for _k in range(1, K + 1):
            if spread_num[user] >= _k:
                idcg_data[user].append(idcg[_k - 1])
            else:
                idcg_data[user].append(idcg_data[user][-1])

    for uid, user in enumerate(test_user_dict):
        algo_rec = sorted(test_user_dict[user], key=lambda x: x[1], reverse=True)
        for _k in range(1, K + 1):

            algo_k = np.array(algo_rec)[:_k, 0]  # get the recommendation list
            n_union = len(np.intersect1d(algo_k, spread_dict[user]))    # compared with the ground truth

            Spread_K_valid[user][_k - 1].extend(np.intersect1d(algo_k, spread_dict[user]).tolist())

            Recall_K[_k - 1].append(n_union / spread_num[user])
            Recall_K_deg[user].append(n_union / spread_num[user])   # calculate the recall@K per user

            if _k == 1:
                DCG_K[_k - 1][uid] = n_union / _k
                NDCG_K[_k - 1].append(DCG_K[_k - 1][uid] / idcg_data[user][_k - 1])
            else:
                is_valid_k = (algo_k[_k - 1] in spread_dict[user])
                DCG_K[_k - 1][uid] = DCG_K[_k - 2][uid] + is_valid_k/np.log2(_k + 1)
                NDCG_K[_k - 1].append(DCG_K[_k - 1][uid] / idcg_data[user][_k - 1])

    # Sperad@K
    Spread_K_full = [[] for _ in range(K)]

    for user in Spread_K_valid:
        for _k in range(K):
            first_order = Spread_K_valid[user][_k].copy()
            second_order = []
            for fst in first_order:
                second_order.extend(spread_dict[fst])
            first_order.extend(second_order)
            Spread_K_full[_k].extend(first_order)

    recall_res = [np.mean(rk) for rk in Recall_K]
    ndcg_res = [np.mean(ndcg) for ndcg in NDCG_K]
    Spread_K_unique = [len(np.unique(spk)) for spk in Spread_K_full]

    if os.path.exists("datasets/ISpread@K.csv") is not True:
        print("Compute the ISpread@K")
        ISpread_K_calculated()

    ISpread_K = np.array(pd.read_csv("data/ISpread@K.csv", header=None)).reshape(-1, )

    print("Recall@1, 2, 3 = [{}, {}, {}]".format(recall_res[0], recall_res[1], recall_res[2]))
    print("NDCG@1, 2, 3 = [{}, {}, {}]".format(ndcg_res[0], ndcg_res[1], ndcg_res[2]))
    print("NSpread@1, 2, 3 = [{}, {}, {}]".format(Spread_K_unique[0]/ISpread_K[0],
                                                  Spread_K_unique[1]/ISpread_K[1],
                                                  Spread_K_unique[2]/ISpread_K[2]))

if __name__ == '__main__':

    EulerNet_data = pd.read_csv("datasets/edge_list.csv").values.tolist()
    Recommendation = []

    with open("data/id2uid.json", "r") as fr:
        idmap = json.load(fr)

    for src, tar, prob in EulerNet_data:
        Recommendation.append([idmap[str(int(src))], idmap[str(int(tar))], prob])

    evaluate(Recommendation)

    pass
