from data_loader import get_psea_data
from scipy.stats import pearsonr,ttest_ind
import numpy as np

ct_feats, pro_feats, labels, ct_feat_names, pathways = get_psea_data()

_, m = ct_feats.shape
_, n = pro_feats.shape
# key_pathways = []
# for i in range(n):
#     positive_samples = [pro_feats[i,j] for j in range(n) if labels[j] == 1]
#     negative_samples = [pro_feats[i,j] for j in range(n) if labels[j] == 0]
#     _, p_value = ttest_ind(positive_samples, negative_samples, equal_var=False)
#     if p_value < 0.1:
#         key_pathways.append(pathways[i])

key_pathways = [
'RNA binding',
'Extracellular exosome',
'Ribonucleoprotein complex',
'Nucleus',
'Protein binding',
'Translation',
'Trna aminoacylation for protein translation',
'Cytoplasm',
'Cytosol',
'Metabolic pathways',
'Oxidoreductase activity, acting on paired donors, with incorporation or reduction of molecular oxygen',
'Mitochondrion'
]

key_pathways = [x.lower() for x in key_pathways]





feats = set()
for i in range(m):
    for j in range(n):
        if pathways[j] in key_pathways:
            correlation_coefficient, pvalue = pearsonr(ct_feats[:,i],pro_feats[:,j])
            if np.abs(correlation_coefficient) > 0.3:
            # if np.abs(pvalue) < 0.01:
                # print(ct_feat_names[i], pathways[j], correlation_coefficient, pvalue)
                feats.add(ct_feat_names[i])

with open("key_features_c0.3.txt","w") as f:
    for feat in feats:
        f.write(feat)
        f.write("\n")
print(len(feats))


