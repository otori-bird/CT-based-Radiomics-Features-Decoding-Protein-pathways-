import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import openpyxl
import os


root_path = "./Kmeans"
Terms2Pval = {}
Terms2Gene = {}
pathways = []
file_path = "Kmeans/david/chart_76698D75AAF61681366275226.txt"
with open(file_path, "r") as f:
    # 读取第一行作为键值
    keys = f.readline().strip().split('\t')
    
    # 逐行读取，每行以空格拆分并存入字典后添加到列表中
    for line in f:
        values = line.strip().split('\t')
        item = {keys[i]: values[i] for i in range(len(keys))}
        
        if '~' in item['Term']:
            func = item['Term'].split('~')[1]
        else:
            func = item['Term'].split(':')[1]
        func = func.lower().strip()
        pathways.append(func)
        Terms2Pval[func] = item['PValue']
        Terms2Gene[func] = [x.strip() for x in item['Genes'].split(',')]


groups = pd.read_csv("../PSEA/PSEA/group_modified.csv",index_col=0)
data = pd.read_excel('../PSEA/PSEA/prot_use.xlsx',index_col=0)

# result_dict = data.to_dict(orient='index')
# for k, v in result_dict.items():
#     result_dict[k] = dict((i, v[i]) for i in data.columns)

# 根据病变等级将病人分组
group_0 = groups[groups['LiverCancerStage'] == 0].index.tolist()
group_1 = groups[groups['LiverCancerStage'] == 1].index.tolist()

for i, cluster in enumerate(pathways):
    try:
        # 读取数据
        genes_subset = Terms2Gene[cluster]
        # 计算每个基因在两个组中的t-test p-value，并计算fold change
        fc_list = []
        pvalue_list = []

        for g in genes_subset:
            fc = data.loc[g, group_1].mean() - data.loc[g, group_0].mean()
            ttest = stats.ttest_ind(data.loc[g, group_1], data.loc[g, group_0])
            pvalue = ttest.pvalue
            fc_list.append(fc)
            pvalue_list.append(pvalue)

                # 绘制火山图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = np.where(np.asarray(fc_list) > 0, 'red', 'blue')
        labels = np.where(np.abs(np.asarray(fc_list)) > 1, genes_subset, '')
        ax.scatter(x=np.asarray(fc_list), y=-1 * np.log10(np.asarray(pvalue_list)), alpha=0.7, s=20, c=np.where(np.abs(np.asarray(fc_list))>1, colors, 'grey'))
        for j in range(len(genes_subset)):
            ax.annotate(labels[j], (np.asarray(fc_list)[j], -1 * np.log10(np.asarray(pvalue_list))[j]), fontsize=8, xytext=(-5,5), textcoords='offset points', ha='center', va='bottom')
        ax.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
        ax.axvline(x=1, color='grey', linestyle='--')
        ax.axvline(x=-1, color='grey', linestyle='--')
        ax.set_xlabel('Fold Change (log2)')
        ax.set_ylabel('-log10 p-value')
        ax.set_title('Volcano Plot')
        file = cluster.replace("/","")
        plt.savefig(f"./Kmeans/david/{file}.png")
    except Exception as e:
        print(e)
        continue