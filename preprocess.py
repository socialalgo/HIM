import json
import pandas as pd
from collections import defaultdict

# src, dst, prob
edge_df = pd.read_csv("your network name.csv")
uid2id = dict()
id2uid = dict()
indeg = defaultdict(int)
outdeg = defaultdict(int)
edge_list = []
id = 0

edges = edge_df.values.tolist()

for row in edges:

    src = int(row[0])
    dst = int(row[1])
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

deg_list = []
for _id in id2uid:
    deg_list.append([_id, outdeg[_id], indeg[_id]])

Node_number = [len(uid2id.keys())]

print("processed!")

with open("id2uid.json", "w") as fp:
    json.dump(id2uid, fp)

df_n = pd.DataFrame(Node_number).to_csv('network_scale.csv', index=False, encoding='utf-8', header=None)
df_deg = pd.DataFrame(deg_list).to_csv('deg_list.csv', index=False, encoding='utf-8', header=None)
df_edge = pd.DataFrame(edge_list).to_csv('edge_list.csv', index=False, encoding='utf-8', header=None)