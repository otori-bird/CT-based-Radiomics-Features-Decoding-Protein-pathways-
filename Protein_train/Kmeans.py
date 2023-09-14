import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.cluster import KMeans
import pickle
import csv
import pandas as pd
from protein_reader import get_protein_data

df = pd.read_excel('../PSEA/PSEA/prot_use.xlsx', usecols=[0], dtype=str)
gene_list = df.iloc[:, 0].tolist()

proteins, genes, _ = get_protein_data()
prot2gene = {x:y for x,y in zip(proteins,genes)}

data_path = "./merged_emb_4919"
fasta_embed_files = os.listdir(data_path)
data = [] 
names = []
layer = 33
data_dict_path = f"./data_dict_4919.pkl"
if not os.path.exists(data_dict_path):
    data0 = []
    data32 = []
    data33 = []
    count = 0 
    for i, f in tqdm(enumerate(fasta_embed_files)):
        prot = f.split("|")[1]
        if prot2gene[prot] not in gene_list:
            continue
        count += 1
        path = os.path.join(data_path, f)
        if os.path.isfile(path):
            fasta_embed_files[i] = path
        else:
            while not os.path.isfile(path):
                path = os.path.join(path, os.listdir(path)[0])
            fasta_embed_files[i] = path
        tmp = torch.load(fasta_embed_files[i])
        data.append(tmp['mean_representations'][layer])
        data0.append(tmp['mean_representations'][0])
        data32.append(tmp['mean_representations'][32])
        data33.append(tmp['mean_representations'][33])
        names.append(tmp['label'].split("|")[1])
    print(count)
    print(len(names))
    x = torch.vstack(data)
    data_dict = {
        "data0":torch.vstack(data0),
        "data32":torch.vstack(data32),
        "data33":torch.vstack(data33),
        "names":names,
    }
    with open(data_dict_path,"wb") as f:
        pickle.dump(data_dict,f)
else:
    with open(data_dict_path,"rb") as f:
        data_dict = pickle.load(f)
    x = data_dict[f'data{layer}']
    names = data_dict['names']

# 使用t-SNE算法进行降维
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(x)

k = 10 # 聚类的簇数
kmeans = KMeans(n_clusters=k).fit(x_tsne) # 训练KMeans模型
labels = kmeans.labels_ # 获取所有样本所属簇的标签


# 根据标签将点拆分成不同的组
groups = {}
for i in range(len(x_tsne)):
    if labels[i] not in groups:
        groups[labels[i]] = []
    groups[labels[i]].append(x_tsne[i])

fig = plt.figure(figsize=(10, 10)) # 设置图像大小为2000x2000像素
# 按组画出散点图
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tan','lightblue','orange']
for i, color in zip(groups.keys(), colors):
    group = np.array(groups[i])
    plt.scatter(group[:, 0], group[:, 1], c=color, label=i)

# plt.legend()
plt.tight_layout()
plt.savefig("kmeans_scatter.png",dpi=300)