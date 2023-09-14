from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_loader import get_split_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
import joblib
import numpy as np
import cvxopt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


# 定义并行处理函数
def train_model_with_parallel(p, roc_auc_flag):
    try:
        X_train = p['X_train']
        X_test = p['X_test']
        y_train = p['y_train']
        y_test = p['y_test']
        param_grid = p['params']
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
            return {
                    "clf":grid_search,
                    "best_params":grid_search.best_params_,
                    "train_score":train_roc_auc,
                    "test_score":test_roc_auc,
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
            return {
                    "clf":grid_search,
                    "best_params":grid_search.best_params_,
                    "train_score":train_acc,
                    "test_score":test_acc,
                    }
    except:
        return None

data_list = [
    "data.xlsx",
]
roc_auc_flag = True
seed = 0
for path in data_list:
    if roc_auc_flag:
        sys.stdout = open(f"roc_auc_{path}.txt","a+")
    else:
        sys.stdout = open(f"acc_{path}.txt","a+")
    print("-----------------------------------------------------")
    print(f"Seed:{seed}")
    # print(f"Processing data file {path}")
    # 划分数据集
    X_train, X_test, y_train, y_test = get_split_data(path=path,seed=seed)

    # 数据预处理，使用StandardScaler标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    models = {
        'SVM':{
            'params':{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'C': [ 0.00001, 0.0001,0.001,0.01, 0.1,],
                        'gamma': [0, 0.00001, 0.0001,0.001, 0.01, 0.1,]},
            'model': SVC(random_state=seed,probability=roc_auc_flag),
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
        },
        'ADA':{
            'params':{
                'n_estimators': [2, 5, 10, 20, 50, 100, 200 ],
                'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
                # 'base_estimator__max_depth': [1, 2, 3, 5],
                # 'base_estimator__min_samples_split': [2, 3, 5]
                },
            'model': AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=seed,max_depth=3,min_samples_leaf=5,criterion='gini'),random_state=seed),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        },
        'DTC':{
            'params':{
                'max_depth': [1, 3, 5, 7],
                'min_samples_leaf': [1, 5, 10, 15],
                'criterion': ['gini', 'entropy','log_loss']
                },
            'model': DecisionTreeClassifier(random_state=seed),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        },
        'GNB':{
            'params':{'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
            'model': GaussianNB(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        },
        'GBT':{
            'params':{'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 500, 1000],
                    'max_depth': [1, 3, 5, 7],
                    'min_samples_leaf': [1, 2, 4]},
            'model': GradientBoostingClassifier(random_state=seed),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
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
                train_score = result['train_score']
                test_score = result['test_score']
                print('Best parameters for', name, ':', clf.best_params_)
                print('Train Score:', train_score)
                print('Test Score:', test_score)
                if roc_auc_flag:
                    print('Test acc', result['test_acc'])
                best_models[name] = {
                    "model": clf.best_estimator_,
                    "params": clf.best_params_,
                    "train_score":train_score,
                    "test_score":test_score,
                }

                if test_score > best_score:
                    best_model = clf.best_estimator_
                    best_score = test_score

    except Exception as e:
            print("Error!")
            print(e)

    finally:
        # 打印每个回归模型的最佳参数和最佳得分
        for model_name, model_info in best_models.items():
            print(f"{model_name}: Best params: {model_info['params']} ")
            print('Train Score:', model_info['train_score'])
            print('Test Score:', model_info['test_score'])

        print('*** Best Model:', best_model)
        print('*** Best Score:', best_score)
