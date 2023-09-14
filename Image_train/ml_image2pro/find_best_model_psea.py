import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from data_loader import get_data, get_psea_data,get_image_data
import warnings
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import joblib



# 屏蔽所有warning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# 定义待选择的超参数范围
param_dist = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4, 5],
    'C': [0.1, 1, 10]
}
ridge_param_dist = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'max_iter' : [10, 50, 100, 500, 1000],
    'tol' : [0.0001, 0.001, 0.01, 0.1],
    'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
}
lasso_param_dict = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
    'fit_intercept': [True, False], 
    'tol': [1e-4, 1e-3, 1e-2],
    'max_iter':[5000,10000],
}
elastic_param_dict = {'alpha': [0.01, 0.1, 1, 10],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
              'tol': [1e-4, 1e-3, 1e-2],
              'max_iter':[5000,10000]}
svr_param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
              'epsilon': [0.01, 0.1, 1, 10],
              'max_iter':[1000,-1]}
dtr_param_grid = {'criterion': ['squared_error', 'absolute_error'], 
                  'max_depth': [2, 4, 6, 8, 10], 
                  'min_samples_split': [2, 5, 10, 15], 
                'min_samples_leaf': [1, 2, 4, 8], 
                'max_features': ['auto', 'sqrt', 'log2', None]}
rfr_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}


# 定义大量回归模型及其对应的超参数范围
models = [
    {'name': 'Linear Regression', 'model': LinearRegression(), 'params': {'fit_intercept': [True, False]}},
    {'name': 'Ridge Regression', 'model': Ridge(), 'params': ridge_param_dist}, # nan?
    {'name': 'Lasso Regression', 'model': Lasso(), 'params': lasso_param_dict}, # more iteration?
    {'name': 'Elastic Net Regression', 'model': ElasticNet(), 'params': elastic_param_dict},  # more iteration?
    {'name': 'SVR', 'model': SVR(), 'params': svr_param_grid}, # acc 0 ???
    {'name': 'Decision Tree Regression', 'model': DecisionTreeRegressor(), 'params': dtr_param_grid},
    {'name': 'Random Forest Regression', 'model': RandomForestRegressor(), 'params': rfr_param_grid}, #
    {'name': 'Adaboost Regression', 'model': AdaBoostRegressor(), 'params': {'n_estimators': [50, 100, 200]}}, # None
    {'name': 'Gradient Boosting Regression', 'model': GradientBoostingRegressor(), 'params': {
        'learning_rate': [0.1, 0.05, 0.01],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }},
    {'name': 'XGBoost Regression', 'model': XGBRegressor(), 'params': {
        'learning_rate': [0.1, 0.05, 0.01],
        'n_estimators': [10, 25, 50, 100, 200],
        'max_depth': [1, 3, 5, 7],
        'min_child_weight': [1, 3, 5,7, 9, 11,]
    }}
]


# 定义并行处理函数
def train_model_with_parallel(reg):
    try:
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2score = r2_score(y_test, y_pred)

        prot_pred = reg.predict(images_feats)
        label_pred = pro2label.predict(prot_pred)
        acc = accuracy_score(labels,label_pred)
        return {'reg': reg, 'acc':acc, 'r2score': r2score}
    except:
        return None

r2score_flag = True

# images_feats, labels = get_image_data()
images_feats, labels = get_image_data()
for split in ['top5','top10']:
    # X, y, labels = get_data()
    # X, y, _ = get_psea_data(feat_path="./total_features.csv",psea_path=f"data_customized_{split}.xlsx")
    X, y, _ = get_psea_data(feat_path="../../myFeatureExtractor/post_train_features.csv",psea_path=f"data_customized_{split}.xlsx")
    if r2score_flag:
        sys.stdout = open(f"r2score_{X.shape[1]}_{y.shape[1]}_{split}_output.txt","a+")
    else:
        sys.stdout = open(f"acc_{X.shape[1]}_{y.shape[1]}_{split}_output.txt","a+")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X, y
    # for model_name in ['ada','svm','gnb']:
    for model_name in ['gbt']:
        pro2label_path = f'../../PSEA/ml/{model_name}_data_customized_{split}.xlsx.joblib'
        # 加载模型
        pro2label = joblib.load(pro2label_path)
        print(f"Pro2Label model:{pro2label_path}")

        # 定义线程池
        executor = ThreadPoolExecutor(max_workers=96)

        # 使用多线程并行处理各回归模型的参数选择任务，并记录最佳得分和对应的最佳模型
        best_model = None
        best_score = 0
        best_acc = 0
        futures = []
        best_models = {}
        try:
            for m in models:
                reg = GridSearchCV(m['model'], m['params'], cv=5, n_jobs=-1)
                # reg = RandomizedSearchCV(m['model'], m['params'], n_iter=10, cv=5)
                future = executor.submit(train_model_with_parallel, reg)
                futures.append((future, m['name']))
            for future, name in futures:
                result = future.result()
                if result is not None:
                    reg = result['reg']
                    acc = result['acc']
                    r2score = result['r2score']
                    print('Best parameters for', name, ':', reg.best_params_)
                    print('Acc on image set:', acc)
                    print('R2 score on test set:', r2score)

                    best_models[name] = {
                        "model": reg.best_estimator_,
                        "params": reg.best_params_,
                        "acc": acc,
                        "r2score": r2score
                    }
                    if r2score_flag:
                        if r2score > best_score:
                            best_model = reg.best_estimator_
                            best_score = r2score
                            best_acc = acc
                    else:
                        if acc > best_acc:
                            best_model = reg.best_estimator_
                            best_score = r2score
                            best_acc = acc
        except Exception as e:
            print("Error!")
            print(e)

        finally:
            # 打印每个回归模型的最佳参数和最佳得分
            for model_name, model_info in best_models.items():
                print(f"{model_name}: Best params: {model_info['params']}, Best score: {model_info['r2score']}, Best acc: {model_info['acc']}")

            print('*** Best Model:', best_model)
            print('*** Best Score:', best_score)
            print('*** Best acc:', best_acc)


