# Experiment 1: C3D + SVM
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
import matplotlib.pylab as plt
from utilities import *
from create_folder import *
t_size=0.7

X, y = get_labels_and_features_from_files(featureBasePath1, True)
#X, y = extractFeatures('weights/weights.h5', '/datarepo/violence-detection-dataset', '', False)


# Cross Validation
clf = svm.SVC(kernel='linear', C = 1, probability=True)

# nsplits = 5
nsplits = 5
#cv = StratifiedKFold(n_splits=nsplits, shuffle=True)
cv = StratifiedShuffleSplit(n_splits=nsplits, train_size=t_size, random_state=33)


tprs = []
aucs = []
scores = []
sens = np.zeros(shape=(nsplits))
specs = np.zeros(shape=(nsplits))
f1Scores = np.zeros(shape=(nsplits))
mean_fpr = np.linspace(0, 1, 100)
plt.figure(num=1, figsize=(7,5))
i = 1
for train, test in cv.split(X, y):
    # train = sklearn.utils.shuffle(train)
    clf.fit(X[train], y[train])
    pred_acc = clf.predict(X[test])
    split_acc = accuracy_score(y[test], pred_acc)
    scores.append(split_acc)
    prediction = clf.predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], prediction[:, 1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC split %d (AUC = %0.4f)' % (i, roc_auc))
    print("ytestBinCount",np.bincount(y[test]))
    print("yBinCount",np.bincount(y))
    print('confusion matrix split ' + str(i))
    print(confusion_matrix(y[test], pred_acc))

    # y_pred = prediction.argmax(axis=-1)
    report = classification_report(y[test], pred_acc, target_names=['ADL', 'Fall'], output_dict=True)
    sens[i - 1] = report['Fall']['recall']
    specs[i - 1] = report['ADL']['recall']
    f1Scores[i - 1] = report['Fall']['f1-score']
    print(classification_report(y[test], prediction.argmax(axis=-1), target_names=['ADL', 'Fall']))
    print('Accuracy: ' + str(split_acc))
    print('\n')
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f %0.4f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)

title=f'Cross-Validation ROC of C3D + SVM model, frame_n:{frame_n}, train size: {t_size}, dataset{zip_f}'
# title = f'Cross-Validation ROC of C3D + SVM model (Variable Value: {variable_value})'
plt.title(title,fontsize=10)
plt.legend(loc="lower right", prop={'size': 10})

plt.savefig('C3D_SVM' + '.pdf')
plt.show()
print("frame_n:",frame_n,"train size:",t_size)
print("dataset",zip_f)
print('Accuracies')
print(scores)
print('Sensitivities')
print(sens)
print('specificities')
print(specs)
print('F1-scores')
print(f1Scores)
print("Avg accuracy: {0} +/- {1}".format(np.mean(scores, axis=0), np.std(scores, axis=0)))
print("Avg sensitivity: {0} +/- {1}".format(np.mean(sens), np.std(sens)))
print("Avg specificity: {0} +/- {1}".format(np.mean(specs), np.std(specs)))
print("Avg f1-score: {0} +/- {1}".format(np.mean(f1Scores), np.std(f1Scores)))
# print("AccAll",split_acc)