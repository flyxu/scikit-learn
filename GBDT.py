# encoding: utf-8

from sklearn import tree
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
from unbalanced_dataset import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
#加载原始libsvm格式数据
data,label=load_svmlight_file("/home/hadoop/input/libsvm.data")
#原始数据集合划分30%作为测试集
x_train,x_test,y_train,y_test=cross_validation.train_test_split(data,label,test_size=0.3,random_state=0)
print np.count_nonzero(y_train==0)
print np.count_nonzero(y_train==1)
#训练数据调用smote算法
verbose = False
ratio = float(np.count_nonzero(y_train==0)) / float(np.count_nonzero(y_train==1))
smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
smox, smoy = smote.fit_transform(x_train.toarray(), y_train)
print np.count_nonzero(smoy==1)
print np.count_nonzero(smoy==0)
#使用决策树模型训练数据
#clf=tree.DecisionTreeClassifier()
clf=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=2,random_state=0)
clf=clf.fit(smox,smoy)
score=clf.score(x_test.toarray(),y_test)
print score
#对测试数据预测
print type(x_test)
print type(x_test.toarray())
y_pred=clf.predict(x_test.toarray())
print y_pred
#模型评估
confusion=confusion_matrix(y_test,y_pred)
print confusion
accruacy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_pred,y_test)
recall = metrics.recall_score(y_test, y_pred)
f1_score=metrics.f1_score(y_test, y_pred)
print accruacy,precision,recall,f1_score
#scores=cross_validation.cross_val_score(clf,data,label,cv=5)#原始数据集合5折交叉验证
#print score
#print scores
#print clf
y_pred_prob = clf.predict_proba(x_test)[:, 1]
auc=metrics.roc_auc_score(y_test, y_pred_prob)
print auc
print cross_validation.cross_val_score(clf, x_test, y_test, scoring='roc_auc').mean()



