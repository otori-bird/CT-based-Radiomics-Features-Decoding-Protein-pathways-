import os
from collections import Counter
from operator import itemgetter
from itertools import groupby
import openpyxl
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import random

prefix = "kegg"
K = 10
topK_list = []

kegg_path = "../kegg.list.xlsx"
go_path = "../go.list.xlsx"
if prefix == "kegg":
    file_path = kegg_path
else:
    file_path = go_path

# 读取Excel工作簿并获取所有的表格名称
workbook = pd.ExcelFile(file_path)
sheet_names = workbook.sheet_names

# 定义一个空列表用于保存结果
all_dictionaries = defaultdict(list)

# 遍历每一个表格
for sheet_name in sheet_names:
    # 读取表格并将第一行作为列名
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    
    # 按照'p.adjust'列的值进行从小到大排序
    df_sorted = df.sort_values(by='p.adjust')
    
    # 提取前十行
    top_rows = df_sorted.iloc[:K]
    
    funcs = []
    for index, row in top_rows.iterrows():
        # dictionary[row['ID']] =  row['geneID']
        all_dictionaries[row['ID']].extend(row['geneID'].split("/"))
        funcs.append(row['ID'])
    topK_list.append(funcs)
    # 将该表格的结果添加到全部结果的列表中
    # all_dictionaries.append(dictionary)
    
for k,v in all_dictionaries.items():
    all_dictionaries[k] = list(set(v))
# 打印全部结果的列表
print(all_dictionaries)



groups = pd.read_csv("../PSEA/PSEA/group_modified.csv",index_col=0)
data = pd.read_excel('../PSEA/PSEA/prot_use_lower.xlsx',index_col=0)


# 根据病变等级将病人分组
group_0 = groups[groups['LiverCancerStage'] == 0].index.tolist()
group_1 = groups[groups['LiverCancerStage'] == 1].index.tolist()

threshold = 7
threshold_fc = 0.5850

os.makedirs(f"./{prefix}{K}_volcanomap",exist_ok=True)
os.makedirs(f"./{prefix}{K}_volcanomap/pathways",exist_ok=True)

selected_gene = pd.DataFrame()
for i, p in enumerate(topK_list):
    os.makedirs(f"./{prefix}{K}_volcanomap/{i}",exist_ok=True)
    for cluster in p:
        file = cluster.replace("/","")
        if os.path.exists(f"./{prefix}{K}_volcanomap/pathways/{file}.txt"):
            continue
        # 读取数据
        genes_subset = all_dictionaries[cluster]
        # 计算每个基因在两个组中的t-test p-value，并计算fold change
        fc_list = []
        pvalue_list = []
        origin_gene_subset = genes_subset.copy()
        genes_subset = []
        for v in origin_gene_subset:
            g = v.lower()
            try:
                fc = np.log2(data.loc[g, group_1].mean() / data.loc[g, group_0].mean())
                # fc = data.loc[g, group_1].mean() - data.loc[g, group_0].mean()
                ttest = stats.ttest_ind(data.loc[g, group_1], data.loc[g, group_0])
                # stats.ttest_1samp
                pvalue = ttest.pvalue
                fc_list.append(fc)
                pvalue_list.append(pvalue)
                genes_subset.append(v)
            except Exception as e:
                print(e)
                continue
        # pvalue_list = fdrcorrection(pvalue_list)[1]

        # 绘制火山图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = np.where(np.asarray(fc_list) > 0, 'red', 'blue')
        labels = np.where(np.abs(np.asarray(fc_list)) > threshold_fc, genes_subset, '')
        ax.scatter(x=np.asarray(fc_list), y=-1 * np.log10(np.asarray(pvalue_list)), alpha=0.7, s=20, c=np.where(np.abs(np.asarray(fc_list))> threshold_fc, colors, 'grey'))
        for j in range(len(genes_subset)):
            ax.annotate(labels[j], (np.asarray(fc_list)[j], -1 * np.log10(np.asarray(pvalue_list))[j]), fontsize=8, xytext=(-5,5), textcoords='offset points', ha='center', va='bottom')
        ax.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
        ax.axvline(x=threshold_fc, color='grey', linestyle='--')
        ax.axvline(x=-threshold_fc, color='grey', linestyle='--')
        ax.set_xlabel('Fold Change (log2)')
        ax.set_ylabel('-log10 p-value')
        ax.set_title('Volcano Plot')
        plt.savefig(f"./{prefix}{K}_volcanomap/{i}/{file}.png")
        
        with open(f"./{prefix}{K}_volcanomap/pathways/{file}.txt","w") as f:
            for g in genes_subset:
                f.write(g)
                f.write("\n")