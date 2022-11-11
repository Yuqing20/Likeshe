import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

# 读取数据
# 训练集
train = pd.read_csv('train.csv')
# 测试集
test = pd.read_csv('test.csv')
# 分离数据集
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
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
    clf = SVR(kernel='rbf', C=1, gamma='scale')
    # 用训练数据拟合分类器模型
    clf.fit(X_train, label_train)
    # 用训练好的分类器去预测测试集的标签
    x1 = clf.predict(X_test)
    # 预测测试集的标签
    y1 = clf.predict(X_test_c)

    # 5折KFold本质上就是把数据集平均切成5份，然后每次选取其中1份作为测试集，
    # 剩下的4份作为训练集来构造模型，重复切5次，每次选取的1/5测试集都不一样,
    # 预测结果也会取5次的平均值，
    prediction1 += y1 / nfold
    i += 1
    # 对给定的数组进行四舍五入 能够指定精度
result1 = np.round(prediction1)
id_ = range(210, 314)
# 创建一个表格 存放结果
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
# 将结果写到csv文件里
df.to_csv("baselineSVM.csv", index=False)
