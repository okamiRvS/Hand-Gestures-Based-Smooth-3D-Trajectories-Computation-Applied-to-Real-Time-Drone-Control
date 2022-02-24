# multi-class classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pdb

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=3, n_features=20, n_informative=3, random_state=42)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# fit model
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
pred_prob = clf.predict_proba(X_test)

# roc curve for classes
fpr = {}
tpr = {}
thresh = {}
roc_auc = {}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

auc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=f'Class 0 vs Rest (area={ round( roc_auc[0], 2) })')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=f'Class 1 vs Rest (area={ round( roc_auc[1], 2) })')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=f'Class 2 vs Rest (area={ round( roc_auc[2], 2) })')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
#plt.savefig('Multiclass ROC',dpi=300);
print(auc_score)
plt.show()