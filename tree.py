# encoding: utf-8

from sklearn import tree
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn import metrics
data,label=load_svmlight_file("/home/hadoop/input/libsvm.data")
#print label
#print data
x_train,x_test,y_train,y_test=cross_validation.train_test_split(data,label,test_size=0.4,random_state=0)#原始数据集合划分40%作为测试集
#print x_train
#print y_train
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)
#score=clf.score(x_test,y_test)
y_pred=clf.predict(x_test)

print metrics.accuracy_score(y_test, y_pred)
print y_pred
confusion=confusion_matrix(y_test,y_pred)
print confusion
TP = confusion[1, 1]
accruacy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
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


