from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from data_loader import get_image_data,get_protein_ct_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve

seed = 0
ct_feats, ct_key_feats, ct_ex_feats, ct_labels = get_image_data(key_feat=True)
protein_ct_feats, protein_key_feats, protein_ex_feats, protein_ct_labels = get_protein_ct_data(key_feat=True)
data_pairs = [
    [ct_feats,ct_labels, protein_ct_feats, protein_ct_labels],
    [ct_key_feats,ct_labels, protein_key_feats, protein_ct_labels],
    [ct_ex_feats,ct_labels, protein_ex_feats, protein_ct_labels],
                ]
plt.figure()
for i, data in enumerate(data_pairs):
    ct_feats, ct_labels, protein_ct_feats, protein_ct_labels = data[0], data[1], data[2], data[3]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(ct_feats, ct_labels, test_size=0.2, random_state=seed)

    # 数据预处理，使用StandardScaler标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    protein_ct_feats = scaler.fit_transform(protein_ct_feats)


    # 定义参数空间
    param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0, 0.00001, 0.0001,0.001,0.01, 0.1,],
                'gamma': [0, 0.00001, 0.0001,0.001, 0.01, 0.1,]}
   
    # 创建SVM模型
    svc = SVC(random_state=42,probability=True)

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(svc, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    clf = grid_search.best_estimator_

    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test, y_test)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    y_prob = clf.predict_proba(protein_ct_feats)[:, 1]
    test_roc_auc = roc_auc_score(protein_ct_labels, y_prob)
    print("Protein ROC_AUC:", test_roc_auc)
    test_acc = clf.score(protein_ct_feats, protein_ct_labels)
    print("Protein Test Accuracy:", test_acc)

    y_prob = clf.predict_proba(X_train)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_prob)
    y_prob = clf.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_prob)
    print("Train ROC_AUC:", train_roc_auc)
    print("Test ROC_AUC:", test_roc_auc)
    fpr, tpr, _ = roc_curve(y_test,y_prob)
    lw = 2

    plt.plot(fpr, tpr, 
            lw=lw, label=f'{i} curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(f"./svm_roc_curve.png")
plt.close()
