from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# 分割训练集和测试集
X, y = train.iloc[:, 0:241].values, train.iloc[:, 241].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
feat_labels = train.columns[0:241]
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
# 下标排序
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    # 给予100颗决策树平均不纯度衰减的计算来评估特征重要性
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
# 可视化特征重要性-依据平均不纯度衰减
plt.title('Feature Importance-RandomForest')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# 选择前100个特征进行分类预测
X_select_pd = train.iloc[:, 0:102]
X_train_c, y_train_c = X_select_pd.values, train.iloc[:, 241].values
X_test_c = test.iloc[:, 0:102].values
nfold = 5
# KFold 是sklearn 包中用于交叉验证的函数。
# 将数据集A分为训练集（training set）B和测试集（test set）C，
# 在样本量不充足的情况下，为了充分利用数据集对算法效果进行测试，
# 将数据集A随机分为k个包，每次将其中一个包作为测试集，剩下k-1个
# 包作为训练集进行训练


# n_splits : 整数，默认为5。表示交叉验证的折数（即将数据集分为几份），
# shuffle : 布尔值, 默认为False。表示是否要将数据打乱顺序后再进行划分。
# random_state :设置random_state=整数，可以保持数据集划分的方式每次都不变，便于不同模型的比较。

kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((len(X_test_c),))
i = 0
for train_index, test_index in kf.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))

    # 加载下载处理好的数据集，然后在将数据集分割成训练集和测试集
    X_train, label_train = X_train_c[train_index], y_train_c[train_index]
    X_test, label_test = X_train_c[test_index], y_train_c[test_index]

    # 使用SVM进行训练
    # 创建分类器对象
    clf = DecisionTreeClassifier()
    # 用训练数据拟合分类器模型
    clf.fit(X_train, label_train)
    # 用训练好的分类器去预测测试集的标签
    x1 = clf.predict(X_test)
    # 预测测试集的标签
    y1 = clf.predict(X_test_c)

    # 5折KFold本质上就是把数据集平均切成5份，然后每次选取其中1份作为测试集，
    # 剩下的4份作为训练集来构造模型，重复切5次，每次选取的1/5测试集都不一样,
    # 预测结果也会取5次的平均值，
    prediction1 += (y1) / nfold
    i += 1

    # 对给定的数组进行四舍五入 能够指定精度
result1 = np.round(prediction1)

id_ = range(210, 314)
# 创建一个表格 存放结果
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
# 将结果写到csv文件里
df.to_csv("baseline特征选择+随机森林.csv", index=False)
