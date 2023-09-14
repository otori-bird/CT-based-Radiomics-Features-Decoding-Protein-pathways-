from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
import joblib
import numpy as np
# import cvxopt
from data_loader import get_image_data, get_protein_ct_data
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor, as_completed

roc_auc_flag = True
ct_feats, ct_key_feats, ct_ex_feats, ct_labels = get_image_data(key_feat=True)
protein_ct_feats, protein_key_feats, protein_ex_feats, protein_ct_labels = get_protein_ct_data(key_feat=True)


# 定义并行处理函数
def train_model_with_parallel(p, roc_auc_flag):
    try:
        X_train = p['X_train']
        X_test = p['X_test']
        y_train = p['y_train']
        y_test = p['y_test']
        param_grid = p['params']
        protein_ct_feats = p['X_protein']
        if roc_auc_flag:
            grid_search = GridSearchCV(p['model'], param_grid, cv=5,scoring='roc_auc',n_jobs=-1)
            grid_search.fit(X_train, y_train)
            # # 打印最佳超参数值
            # print("Best parameters: {}".format(grid_search.best_params_))
            # # 使用最佳超参数拟合模型
            clf = grid_search.best_estimator_
            # 预测概率值
            y_prob = clf.predict_proba(X_train)[:, 1]
            train_roc_auc = roc_auc_score(y_train, y_prob)
            y_prob = clf.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_prob)
            test_acc = clf.score(X_test, y_test)

            # print("Train ROC_AUC:", train_roc_auc)
            # print("Test ROC_AUC:", test_roc_auc)

            y_prob = clf.predict_proba(protein_ct_feats)[:, 1]
            protein_roc_auc = roc_auc_score(protein_ct_labels,y_prob)
            # print("Protein CT ROC_AUC:", test_roc_auc)
            return {
                    "clf":grid_search,
                    "X_test":X_test,
                    "y_test":y_test,
                    "best_params":grid_search.best_params_,
                    "train_score":train_roc_auc,
                    "test_score":test_roc_auc,
                    "protein_score":protein_roc_auc,
                    "test_acc": test_acc,
                    }
        else:
            grid_search = GridSearchCV(p['model'], param_grid, cv=5,n_jobs=-1)
            grid_search.fit(X_train, y_train)
            # 打印最佳超参数值
            # print("Best parameters: {}".format(grid_search.best_params_))
            # 使用最佳超参数拟合模型
            clf = grid_search.best_estimator_
            # 计算准确率
            train_acc = clf.score(X_train,y_train)
            test_acc = clf.score(X_test, y_test)
            # print("Train Accuracy:", train_acc)
            # print("Test Accuracy:", test_acc)
            protein_acc = clf.score(protein_ct_feats, protein_ct_labels)
            # print("Protein CT Accuracy:", test_acc)
            return {
                    "clf":grid_search,
                    "best_params":grid_search.best_params_,
                    "train_score":train_acc,
                    "test_score":test_acc,
                    "protein_score":protein_acc
                    }
    except:
        return None



if roc_auc_flag:
    sys.stdout = open(f"triple_svm_roc_auc_{protein_ct_feats.shape[1]}_output.txt","a+")
else:
    sys.stdout = open(f"acc_{protein_ct_feats.shape[1]}_output.txt","a+")
data_pairs = [
    [ct_feats,ct_labels, protein_ct_feats, protein_ct_labels],
    [ct_key_feats,ct_labels, protein_key_feats, protein_ct_labels],
    [ct_ex_feats,ct_labels, protein_ex_feats, protein_ct_labels],
                ]

fig = plt.figure(figsize=(10,10))

legends = [
    "All",
    "Key",
    "Remaining"
]
seed = 0
sys.stdout.flush()
print("-----------------------------------------------------")
print(f"Seed:{seed}")
for i, data in enumerate(data_pairs):
	ct_feats, ct_labels, protein_ct_feats, protein_ct_labels = data[0], data[1], data[2], data[3]

	# 划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(ct_feats, ct_labels, test_size=0.2, random_state=seed)

	# 数据预处理，使用StandardScaler标准化数据
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)
	protein_ct_feats_scaled = scaler.fit_transform(protein_ct_feats)

	models = {
		'SVM':{
			'params':{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
						'C': [ 0.00001, 0.0001,0.001,0.01, 0.1,],
						'gamma': [0, 0.00001, 0.0001,0.001, 0.01, 0.1,]},
			'model': SVC(random_state=42,probability=roc_auc_flag),
			'X_train': X_train_scaled,
			'X_test': X_test_scaled,
			'y_train': y_train,
			'y_test': y_test,
			'X_protein': protein_ct_feats_scaled,
		},
	}

	# 定义线程池
	executor = ThreadPoolExecutor(max_workers=96)
	# 使用多线程并行处理各回归模型的参数选择任务，并记录最佳得分和对应的最佳模型
	best_model = None
	best_score = 0
	futures = []
	best_models = {}
	
	try:
		for key,p in models.items():
			future = executor.submit(train_model_with_parallel, p, roc_auc_flag)
			futures.append((future, key))
		for future, name in futures:
			result = future.result()
			if result is not None:
				clf = result['clf']
				X_test = result['X_test']
				y_test = result['y_test']
				y_prob = clf.best_estimator_.predict_proba(X_test)[:, 1]
				test_roc_auc = roc_auc_score(y_test, y_prob)
				fpr, tpr, _ = roc_curve(y_test,y_prob)
				lw = 4

				plt.plot(fpr, tpr, 
						lw=lw, label=f'{legends[i]} (AUC=%0.2f)' % test_roc_auc)

				train_score = result['train_score']
				test_score = result['test_score']
				protein_score = result['protein_score']
				print('Best parameters for', name, ':', clf.best_params_)
				print('Train Score:', train_score)
				print('Test Score:', test_score)
				print('Protein Score:', protein_score)
				if roc_auc_flag:
					print('Test acc', result['test_acc'])

				best_models[name] = {
					"model": clf.best_estimator_,
					"params": clf.best_params_,
					"train_score":train_score,
					"test_score":test_score,
					"protein_score":protein_score
				}

				if test_score > best_score:
					best_model = clf.best_estimator_
					best_score = test_score

	except Exception as e:
			print("Error!")
			print(e)

	finally:
		# 打印每个回归模型的最佳参数和最佳得分
		# for model_name, model_info in best_models.items():
		#     print(f"{model_name}: Best params: {model_info['params']} ")
		#     print('Train Score:', model_info['train_score'])
		#     print('Test Score:', model_info['test_score'])
		#     print('Protein Score:', model_info['protein_score'])

		print('*** Best Model:', best_model)
		print('*** Best Score:', best_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('1 - Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.title('Receiver operating characteristic',fontsize=20)
plt.legend(loc="lower right",fontsize=20)
plt.tight_layout()
plt.savefig(f"./svm_roc_curve.svg")
plt.close()
