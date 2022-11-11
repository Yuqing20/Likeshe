from lightgbm import log_evaluation, early_stopping
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.svm import SVR

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# 分割训练集和测试集
X, y = train.iloc[:, 0:241].values, train.iloc[:, 241].values
X, y = train.iloc[:, 0:241].values, train.iloc[:, 241].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
feat_labels = train.columns[0:241]
forest = RandomForestClassifier(n_estimators=30000, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
# 特征重要性
importances = forest.feature_importances_
# 下标排序
indices = np.argsort(importances)[::-1]  # 逆序输出
for f in range(X_train.shape[1]):
    # 给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
# 可视化特征重要性-依据平均不纯度衰减
plt.title('Feature Importance-RandomForest')  # 标题
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')  # 画柱状图
# 获取或设置当前x轴刻度位置和标签
# ticks：x轴刻度位置的列表
# labels：放在指定刻度位置的标签文本。
# ** kwargs：文本属性用来控制标签文本的展示，例如字体大小、字体样式等
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
# 显示x轴的作图范围
plt.xlim([-1, X_train.shape[1]])
# 使图像填满整个区域
plt.tight_layout()
plt.show()

# 选择前100个特征
X = train.iloc[:, 1:102].astype(float)
Y = train.iloc[:, 241]
X_test = test.iloc[:, 1:102].astype(float)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

num_folds = 10
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test))
auc_score = 0

params = {'num_leaves': int(16),
          'objective': 'regression',
          'max_depth': int(4),
          'min_data_in_leaf': int(5),
          'min_sum_hessian_in_leaf': int(0),
          'learning_rate': 0.18,
          'boosting': 'gbdt',
          'feature_fraction': 0.8,
          'bagging_freq': int(2),
          'bagging_fraction': 1,
          'bagging_seed': 8,
          'lambda_l1': 0.01,
          'lambda_l2': 0.01,
          'metric': 'auc',  ##评价函数选择
          "random_state": 2020,  # 随机数种子，可以防止每次运行的结果不一致
          "verbose": -1
          }
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, Y)):
    print("Fold: ", fold_ + 1)
    X_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], Y.iloc[val_idx]
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_valid, y_valid, reference=trn_data)

    clf = lgb.train(params,
                    trn_data,
                    10,
                    valid_sets=val_data,
                    callbacks=callbacks
                    )
    y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    auc = roc_auc_score(y_valid, y_pred)
    print(auc)
    auc_score += auc

    preds = clf.predict(X_test, num_iteration=clf.best_iteration)
    test_result += preds
auc_score = auc_score / folds.n_splits
print("AUC score: ", auc_score)
test_result = test_result / folds.n_splits
Y_test = np.round(test_result)

id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': Y_test})
df.to_csv("baselinelgb+特征选择.csv", index=False)
