import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from lightgbm import log_evaluation, early_stopping

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

# 数据加载
dataframe = pd.read_csv("train.csv")
dataframe_test = pd.read_csv("test.csv")
X = dataframe.iloc[:, 1:241].astype(float)
Y = dataframe.iloc[:, 241]
X_test = dataframe_test.iloc[:, 1:241].astype(float)

num_folds = 8
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test))
auc_score = 0

params = {'num_leaves': int(16),
          'objective': 'regression',
          'max_depth': int(4),  # 最大深度
          'min_data_in_leaf': int(5),
          'min_sum_hessian_in_leaf': int(0),
          'learning_rate': 0.18,  # 每一步迭代的步长
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
test_result = test_result / folds.n_splits
Y_test = np.round(test_result)

id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': Y_test})
df.to_csv("baselinelgb.csv", index=False)
# 评价指标
print('LightGBM Model accuracy score:{0:0.4f}'.format(roc_auc_score(y_valid, y_pred)))
