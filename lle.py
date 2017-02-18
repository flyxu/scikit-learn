from sklearn import manifold,datasets
import pandas as pd
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import  cross_val_score
from unbalanced_dataset import SMOTE
import numpy as np

train=pd.read_csv('./cmv.csv')
train['Defective']=train['Defective'].map({'Y':1,'N':0})
print type(train.values)
train=train.values
print train[0:1]
X_r,err=manifold.locally_linear_embedding(train[:,0:-1],n_neighbors=12,n_components=4)
print("Done. Reconstruction error: %g" % err)
data=X_r
label=train[:,-1]
#print label
x_train,x_test,y_train,y_test=cross_validation.train_test_split(data,label,test_size=0.3,random_state=0)
verbose = False
ratio = float(np.count_nonzero(y_train==0)) / float(np.count_nonzero(y_train==1))
smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
smox, smoy = smote.fit_transform(x_train, y_train)
print np.count_nonzero(smoy==1)
print np.count_nonzero(smoy==0)
clf=svm.SVC(C=10000,gamma=0.0078125)
#print y_train.astype(int)
clf.fit(smox,smoy)
y_pred=clf.predict(x_test)
print y_test
print y_pred
confusion=confusion_matrix(y_test,y_pred)
print confusion
score = cross_val_score(clf, x_train, y_train)
print score.mean()
print score.std()
