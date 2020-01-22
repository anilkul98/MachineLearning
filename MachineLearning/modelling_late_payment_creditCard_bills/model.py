import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import svm
warnings.filterwarnings(action="ignore")

def preprocessData(x,x_test,y):
    ratio=round((y[y['TARGET']==0].shape[0]) / (y[y['TARGET']==1].shape[0]))
    x["TARGET"] = y.TARGET
    duplicate = x[x["TARGET"] == 1]
    x = x.append([duplicate] * ratio, ignore_index=True)
    y = x.TARGET
    x = x.drop("TARGET", axis=1)

    x.drop(x.columns[x.isna().sum() > len(x) * 0.3], axis=1)
    x = x.fillna(x.mean())
    x = x.fillna(x.mode())
    nonNumericCols = x.dtypes[(x.dtypes != int) & (x.dtypes != float)].index
    x = pd.get_dummies(x, nonNumericCols)

    x_test.drop(x_test.columns[x_test.isna().sum() > len(x_test) * 0.3], axis=1)
    x_test = x_test.fillna(x_test.mean())
    x_test = x_test.fillna(x_test.mode())
    nonNumericCols = x_test.dtypes[(x_test.dtypes != int) & (x_test.dtypes != float)].index
    x_test = pd.get_dummies(x_test, nonNumericCols)

    missing_cols = set(x.columns) - set(x_test.columns)
    for c in missing_cols:
        x_test[c] = 0
    missing_cols_test = set(x_test.columns) - set(x.columns)
    for c in missing_cols_test:
        x[c] = 0
    pca = PCA(n_components=20, whiten=False, random_state=2019)
    x = pca.fit(x).transform(x)
    x_test = pca.transform(x_test)
    return x,x_test,y

def applyModel(model,X_train,y_train,X_test,i,id):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(model.__str__().split("(")[0])
    print("--------------------")
    df = pd.DataFrame({'ID': id, 'TARGET': y_pred})
    df.to_csv("target_truth{}".format(i),index=False)
    print("AUROC_TRAIN: {}".format(roc_auc_score(y_train, y_pred_train)))

def readData(i):
    x = pd.read_csv("hw07_target{}_training_data.csv".format(i), index_col=0)
    y = pd.read_csv("hw07_target{}_training_label.csv".format(i), index_col=0)
    x_test = pd.read_csv("hw07_target{}_test_data.csv".format(i))
    id = x_test['ID']
    x_test.drop('ID', axis=1, inplace=True)
    X_train, X_test, y_train = preprocessData(x, x_test, y)
    return id, X_train, X_test, y_train

models=[svm.SVC()]

for i in range (3):
    print("TARGET{}:".format(i+1))
    print("--------")
    id, X_train, X_test, y_train = readData(i+1)

    for m in models:
        applyModel(models[0], X_train, y_train, X_test, 1, id)