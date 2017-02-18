import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets,feature_selection,cross_validation
from sklearn.pipeline import Pipeline

digits=datasets.load_digits()
#print digits.data
y=digits.target
y=y[:200]
x=digits.data[:200]
n_samples=len(y)
print x.shape
x=x.reshape((n_samples,-1))
x=np.hstack((x,2*np.random.random((n_samples,200))))
print x.shape
transform=feature_selection.SelectPercentile(feature_selection.f_classif)
clf=Pipeline([('anova',transform),('svc',svm.SVC(C=1.0))])
score_means=list()
score_stds=list()
percentiles=(1,3,6,10,15,20,30,40,60,80,100)
for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores=cross_validation.cross_val_score(clf,x,y,n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())
print score_means
print score_stds
plt.errorbar(percentiles,score_means,np.array(score_stds))
plt.title('Performance of the SVM-Anova varying the percentile of feature selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')
plt.axis('tight')
plt.show()