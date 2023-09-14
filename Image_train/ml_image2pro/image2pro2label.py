import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from data_loader import get_data, get_psea_data, get_image_data
import warnings
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib


# 加载模型
pro2label = joblib.load('svm_data_customized_15.xlsx.joblib')

# 屏蔽所有warning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
X, y, _ = get_psea_data()
images_feats, labels = get_image_data()

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# 定义大量回归模型及其对应的超参数范围
models = [
    {'name': 'Linear Regression', 'model': LinearRegression(fit_intercept=False)},
    {'name': 'Ridge Regression', 'model': Ridge(alpha=0.01, max_iter=10, solver='sag', tol=0.001)},
    {'name': 'Lasso Regression', 'model': Lasso(alpha=100, fit_intercept=False, tol=0.0001)},
    {'name': 'Elastic Net Regression', 'model': ElasticNet(alpha=0.01, l1_ratio=0.1, tol=0.0001)},
    # {'name': 'SVR', 'model': SVR(), 'params': svr_param_grid},
    {'name': 'Decision Tree Regression', 'model': DecisionTreeRegressor(max_depth=3),},
    {'name': 'Random Forest Regression', 'model': RandomForestRegressor(n_estimators=100)},
    # {'name': 'Adaboost Regression', 'model': AdaBoostRegressor(), 'params': {'n_estimators': [50, 100, 200]}},
    # {'name': 'Gradient Boosting Regression', 'model': GradientBoostingRegressor(), 'params': {
        # 'learning_rate': [0.1, 0.05, 0.01],
        # 'n_estimators': [50, 100, 200],
        # 'max_depth': [3, 5, 7],
        # 'min_samples_split': [2, 5, 10]
    # }},
    {'name': 'XGBoost Regression', 'model': XGBRegressor(learning_rate=0.01, max_depth= 5, min_child_weight=5, n_estimators=50)}
]


# 使用多线程并行处理各回归模型的参数选择任务，并记录最佳得分和对应的最佳模型
best_model = None
best_score = 0
best_models = {}
try:
    for m in models:
        reg = m['model']
        reg.fit(X_train, y_train)

        prot_pred = reg.predict(images_feats)
        label_pred = pro2label.predict(prot_pred)
        score = accuracy_score(labels,label_pred)
        if score > best_score:
            best_model = reg
            best_score = score
except Exception as e:
    print(e)

finally:
    # 打印每个回归模型的最佳参数和最佳得分
    for model_name, model_info in best_models.items():
        print(f"{model_name}: Best params: {model_info['params']}, Best score: {model_info['score']}")

    print('*** Best Model:', best_model)
    print('*** Best Score:', best_score)


