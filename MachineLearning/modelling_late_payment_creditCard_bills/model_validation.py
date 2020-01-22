import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def preprocessData(x,y):
    x.drop(['ID'], axis=1, inplace=True)
    x.drop(x.columns[x.isna().sum() > len(x) * 0.3], axis=1)
    x = x.fillna(x.mean())
    x = x.fillna(x.mode())
    nonNumericCols = x.dtypes[(x.dtypes != int) & (x.dtypes != float)].index
    x = pd.get_dummies(x, nonNumericCols)
    pca = PCA(n_components=20, whiten=False, random_state=2019)
    x=pca.fit_transform(x)
    return x

x=pd.read_csv("hw07_target1_training_data.csv")
y=pd.read_csv("hw07_target1_training_label.csv", index_col=0)
x.insert(0,'TARGET',y)
x_processed = preprocessData(x,y)
X_train, X_test, y_train, y_test = train_test_split(x_processed,y, test_size= 0.1, random_state=16)

print("RANDOM FOREST MODEL:")
print("--------------------")
model1 = RandomForestClassifier(n_estimators=100, random_state=16)
model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
print(y_pred)
print("AUROC: {}".format(roc_auc_score(y_test, y_pred)))

print("SVM MODEL:")
print("--------------------")
model2 = svm.SVC()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)
print(y_pred)
print("AUROC: {}".format(roc_auc_score(y_test, y_pred)))

print("ADA BOOST MODEL:")
print("--------------------")
model3 = AdaBoostClassifier(n_estimators=100, random_state=0)
model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)
print(y_pred)
print("AUROC: {}".format(roc_auc_score(y_test, y_pred)))


print("XGBOOST MODEL:")
print("--------------------")
model4 = XGBClassifier()
model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)
print(y_pred)
print("AUROC: {}".format(roc_auc_score(y_test, y_pred)))

print("NAIVE BAYES MODEL:")
print("--------------------")
model5 = GaussianNB()
model5.fit(X_train, y_train)

y_pred = model5.predict(X_test)
y_pred_train = model5.predict(X_train)
print(y_pred)
print("AUROC_TEST: {}".format(roc_auc_score(y_test, y_pred)))
print("AUROC_TRAIN: {}".format(roc_auc_score(y_train, y_pred_train)))
print(confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_train, y_pred_train))


print("VOTING CLASSIFIER MODEL:")
print("--------------------")
model6 = VotingClassifier(estimators=[('xgb', model4), ('nb', model5)], voting='hard')
model6.fit(X_train, y_train)

y_pred = model6.predict(X_test)
print(y_pred)
print("AUROC: {}".format(roc_auc_score(y_test, y_pred)))
