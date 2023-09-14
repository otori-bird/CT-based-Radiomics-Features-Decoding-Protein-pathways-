import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import statsmodels

# 读取基因表达谱数据
df = pd.read_excel('../PSEA/PSEA/prot_use.xlsx',index_col=0)
groups = pd.read_csv("../PSEA/PSEA/group_modified.csv",index_col=0)

group_0 = groups[groups['LiverCancerStage'] == 0].index.tolist()
group_1 = groups[groups['LiverCancerStage'] == 1].index.tolist()

# 将数据集分成阳性和阴性两组
all_cols = set(df.columns)
cols_to_drop = list(all_cols - set(group_0))
df_dropped = df.drop(columns=cols_to_drop)
negative_data = df_dropped.loc[:, group_0]

cols_to_drop = list(all_cols - set(group_1))
df_dropped = df.drop(columns=cols_to_drop)
positive_data = df_dropped.loc[:, group_1]

# 计算每个基因在阳性和阴性两组之间的差异
mean_positive = positive_data.mean(axis=1)
mean_negative = negative_data.mean(axis=1)
std_positive = positive_data.std(axis=1)
std_negative = negative_data.std(axis=1)

# 进行t-test或ANOVA等统计检验，得到raw p-value
p_values = []
for i in range(len(df)):
    _, pvalue = stats.ttest_ind(positive_data.iloc[i, :], negative_data.iloc[i,:]) # 对每个基因进行t-test
    p_values.append(pvalue)


# 对p-value进行FDR方法矫正
# p_values_corrected = statsmodels.stats.multitest.multipletests(p_values, method='fdr_bh')[1]

# 计算fold change
# fold_change = np.log2(mean_positive / mean_negative)
fold_change = mean_positive - mean_negative

# 绘制火山图
fig, ax = plt.subplots()
ax.scatter(fold_change, -np.log10(p_values), alpha=0.5, color='black')
plt.xlabel('Fold Change')
plt.ylabel('-log10(Adjusted p-value)')
plt.title('Volcano Plot')

plt.savefig(f"./Kmeans/all_gene_vocalno.png")
            
