# -*- coding: utf-8 -*-
#!/bin/python

import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score




data = pd.read_csv("census.csv")

# TODO：总的记录数
n_records = data.shape[0];
# TODO：被调查者的收入大于$50,000的人数
#c = np.loc([income["income"]])
greater = data[data["income"] == ">50K"]
n_greater_50k = greater.shape[0]

#print n_greater_50k
# TODO：被调查者的收入最多为$50,000的人数
most = data[data["income"] == "<=50K"]
n_at_most_50k = most.shape[0]

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = n_greater_50k / float(n_records)

# 打印结果
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 导入sklearn.preprocessing.StandardScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
print features_raw.head(n = 1)

# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = pd.get_dummies(income_raw)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# 移除下面一行的注释以观察编码的特征名字
# print encoded

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# 显示切分的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

print ""
print "----------------- naive classifier ----------------"

# TODO： 计算准确率
accuracy = float(n_greater_50k) / (n_greater_50k + n_at_most_50k)

# when its predict yes ,how offen its correct
precision = float(n_greater_50k) / (n_greater_50k + n_at_most_50k)
#true positive rate
recall = float(n_greater_50k) / n_greater_50k
# TODO： 使用上面的公式，并设置beta=0.5计算F-score
fscore = 1.5 * (precision * recall) / ((0.25 * precision) + recall)

# 打印结果
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)




# 这个代码写得让人摸不着头脑，为啥这么多参数？ bad taste
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    print "train set shape : {}".format(X_train.shape)
    
    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # 获得程序开始时间
    
    X_train = X_train[:sample_size - 1]
    y_train = y_train[:sample_size - 1]
    
    
    learner = learner.fit(X_train, y_train)
    end = time() # 获得程序结束时间
    
    # TODO：计算训练时间
    results['train_time'] = end - start
    
    # TODO: 得到在测试集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # 获得程序结束时间
    
    # TODO：计算预测用时
    results['pred_time'] = end - start
            
    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train, predictions_train)
        
    # TODO：计算在测试集上的准确率
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train, predictions_train, 1)
        
    # TODO：计算测试集上的F-score
    results['f_test'] = fbeta_score(y_test, predictions_test, 1)
       
    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # 返回结果
    return results


# TODO：从sklearn中导入三个监督学习模型
from sklearn import __version__
print __version__


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# TODO：初始化三个模型
clf_A = RandomForestClassifier()
clf_B = SVC()
clf_C = LogisticRegression()

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
trainSize = X_train.shape[0]
samples_1 = trainSize / 100
samples_10 = trainSize / 10
samples_100 = trainSize

# 收集学习器的结果
results = {}
for clf in [clf_B]:
    clf_name = clf.__class__.__name__
    print clf_name
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        #print "trainning with {} columns data.".format(samples)

        #if clf_name == "SVC" or clf_name == "LogisticRegression":
        #    y_train4predict = y_train['>50K']
        #    y_test4predict = y_test['>50K']
        #else :
        y_train4predict = y_train['>50K']
        print(y_train4predict)
        y_test4predict = y_test['>50K']
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train4predict, X_test, y_test4predict)
        #train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
