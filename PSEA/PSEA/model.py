import numpy  as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data  = pd.read_excel("data.xlsx")
group = data.pop("group")
X = np.array(data)
y = np.array(group)

scaler  = preprocessing.StandardScaler().fit(X)
X_scale = scaler.transform(X)

seed = 0
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, stratify=y, random_state=seed) # test_size 测试集占比
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

predict_y_tra = model.predict_proba(X_train)
predict_y_te  = model.predict_proba(X_test)
auc_tra       = roc_auc_score(y_true=y_train, y_score=predict_y_tra[:,1])
auc_te        = roc_auc_score(y_true=y_test, y_score=predict_y_te[:,1])
print("seed = {}".format(seed))
print("LM:: AUC on train Cohort: {}.".format(auc_tra))
print("LM:: AUC on test Cohort: {}.".format(auc_te))

rf = RandomForestClassifier(n_estimators=1, max_depth=5,min_samples_split=2, random_state=1)
rf.fit(X_train, y_train)
predict_y_tra_rf = rf.predict_proba(X_train)
predict_y_te_rf  = rf.predict_proba(X_test)
auc_tra_rf       = roc_auc_score(y_true=y_train, y_score=predict_y_tra_rf[:,1])
auc_te_rf        = roc_auc_score(y_true=y_test, y_score=predict_y_te_rf[:,1])
print("RF:: AUC on train Cohort: {}.".format(auc_tra_rf))
print("RF:: AUC on test Cohort: {}.".format(auc_te_rf))
print("-----------------------------------------------------")

