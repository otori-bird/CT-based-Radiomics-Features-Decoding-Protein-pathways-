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

random.seed(33)

def pathway_condition(item):
    # if float(item['PValue']) > 1e-5:
    #     return False
    if float(item['Bonferroni']) > 0.05:
        return False
    if float(item['FDR']) > 0.05:
        return False
    return True

topk_flag = True

if topk_flag:
    prefix = "esm_"
else:
    prefix = "random"

K = 10

root_path = "./Kmeans"
# root_path = "./random_categorys"
categories = []
functions = []
Terms2Pval = []
Terms2FDR = []
Term2Per = []
Terms2Gene = defaultdict(list)
topK_list = []

print("请告诉我以下生物学名词与肝癌（HCC）联系最为紧密、对肝癌的发现、预防、治疗最有帮助的")
for subdir, dirs, files in os.walk(root_path):
    for file in files:
        if file.startswith("chart"):
            file_path = os.path.join(subdir, file)
            pathways = []
            funcs = []
            t2p = {}
            t2fdr = {}
            t2per = {}
            with open(file_path, "r") as f:
                # 读取第一行作为键值
                keys = f.readline().strip().split('\t')
                
                # 逐行读取，每行以空格拆分并存入字典后添加到列表中
                for line in f:
                    values = line.strip().split('\t')
                    item = {keys[i]: values[i] for i in range(len(keys))}
                    if not pathway_condition(item):
                        continue
                    pathways.append(item)
                    if '~' in item['Term']:
                        func = item['Term'].split('~')[1].lower()
                    else:
                        func = item['Term'].split(':')[1].lower()
                        
                    # 调用 Translator 对象的 translate() 方法进行翻译
                    # func = translator.translate(func, dest='zh-cn').text
                    funcs.append(func)
                    t2p[func] = item['Bonferroni']
                    t2fdr[func] = item['FDR']
                    t2per[func] = item['%']
                    Terms2Gene[func].extend([x.strip() for x in item['Genes'].split(',')])
                
            categories.append(pathways)
            functions.append(set(funcs))
            Terms2Pval.append(t2p)
            Term2Per.append(t2per)
            Terms2FDR.append(t2fdr)

            if topk_flag:
                final_list = sorted(sorted(funcs, key=lambda x: float(t2p[x])), key=lambda x: float(t2fdr[x]))[:K]
            else:
                final_list = random.sample(funcs,K)
            for x in final_list:
                print(x)
            topK_list.append(final_list)
            
print(len(categories))


groups = pd.read_csv("../PSEA/PSEA/group_modified.csv",index_col=0)
data = pd.read_excel('../PSEA/PSEA/prot_use_lower.xlsx',index_col=0)

# result_dict = data.to_dict(orient='index')
# for k, v in result_dict.items():
#     result_dict[k] = dict((i, v[i]) for i in data.columns)

# 根据病变等级将病人分组
group_0 = groups[groups['LiverCancerStage'] == 0].index.tolist()
group_1 = groups[groups['LiverCancerStage'] == 1].index.tolist()

threshold = 7
threshold_fc = 0.5850

os.makedirs(f"./{prefix}{K}_volcanomap",exist_ok=True)
os.makedirs(f"./{prefix}{K}_volcanomap/pathways",exist_ok=True)

key_genes = set()
selected_gene = pd.DataFrame()
for i, p in enumerate(topK_list):
    os.makedirs(f"./{prefix}{K}_volcanomap/{i}",exist_ok=True)
    for cluster in p:
        file = cluster.replace("/","")
        if file not in ['nucleus','cytoplasm','protein binding']:
            continue
        # if os.path.exists(f"./{prefix}{K}_volcanomap/pathways/{file}.txt"):
            # continue
        # 读取数据
        genes_subset = Terms2Gene[cluster]
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
        labels = np.where((np.abs(np.asarray(fc_list)) > threshold_fc) & (np.asarray(pvalue_list) < 0.05),genes_subset, '')
        ax.scatter(x=np.asarray(fc_list), y=-1 * np.log10(np.asarray(pvalue_list)), alpha=0.7, s=20, c=np.where((np.abs(np.asarray(fc_list)) > threshold_fc) & (np.asarray(pvalue_list) < 0.05), colors, 'grey'))
        for j in range(len(genes_subset)):
            key_genes.add(labels[j])
            ax.annotate(labels[j], (np.asarray(fc_list)[j], -1 * np.log10(np.asarray(pvalue_list))[j]), fontsize=8, xytext=(-5,5), textcoords='offset points', ha='center', va='bottom')
        ax.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
        ax.axvline(x=threshold_fc, color='grey', linestyle='--')
        ax.axvline(x=-threshold_fc, color='grey', linestyle='--')
        ax.set_xlabel('Fold Change (log2)',fontsize=20)
        ax.set_ylabel('-log10 p-value',fontsize=20)
        ax.set_title('Volcano Plot',fontsize=20)
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.tight_layout()
        plt.savefig(f"{file}.svg")
        
        # with open(f"./{prefix}{K}_volcanomap/pathways/{file}.txt","w") as f:
        #     for g in genes_subset:
        #         f.write(g)
        #         f.write("\n")
# with open(f"./{prefix}{K}_volcanomap/key_genes.txt","w") as f:
#     for g in key_genes:
#         f.write(g)
#         f.write("\n")