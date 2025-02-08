import pandas as pd
from collections import defaultdict

edge_df = pd.read_csv("twitter_EulerNet_new_normed.csv")
uid2id = dict()
id2uid = dict()
indeg = defaultdict(int)
outdeg = defaultdict(int)
ap = set()
edge_list = []
id = 0

edges = edge_df.values.tolist()

for row in edges:

    src = row[0]
    dst = row[1]
    prob = row[2]
    if src not in uid2id:
        id2uid[id] = src
        uid2id[src] = id
        id = id + 1
    if dst not in uid2id:
        id2uid[id] = dst
        uid2id[dst] = id
        id = id + 1

    edge_list.append([uid2id[src], uid2id[dst], prob])
    indeg[uid2id[dst]] = indeg[uid2id[dst]] + 1
    outdeg[uid2id[src]] = outdeg[uid2id[src]] + 1
    ap.add(uid2id[src])

deg_list = []
for _id in id2uid:
    deg_list.append([_id, outdeg[_id], indeg[_id]])

Node_number = [len(uid2id.keys())]

ap_list = list(ap)

print("processed!")

df_n = pd.DataFrame(Node_number).to_csv('network_scale.csv', index=False, encoding='utf-8', header=None)
df_deg = pd.DataFrame(deg_list).to_csv('deg_list.csv', index=False, encoding='utf-8', header=None)
df_ap = pd.DataFrame(ap_list).to_csv('ap_list.csv', index=False, encoding='utf-8', header=None)
df_edge = pd.DataFrame(edge_list).to_csv('edge_list.csv', index=False, encoding='utf-8', header=None)