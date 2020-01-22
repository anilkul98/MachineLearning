import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
warnings.filterwarnings(action="ignore")


def preprocessData(x,x_test,y,i):
    ratio=round((y[y==0].shape[0]) / (y[y==1].shape[0]))
    x['TARGET'] = y
    duplicate = x[x['TARGET'] == 1]
    x = x.append([duplicate] * ratio, ignore_index=True)
    y = x.TARGET
    x = x.drop('TARGET', axis=1)

    x = x.drop(x.columns[x.isna().sum() > len(x) * 0.3], axis=1)
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
    return x, x_test, y

def applyModel(model,X_train,y_train,X_test,i):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(model.__str__().split("(")[0])
    print("--------------------")
    pd.DataFrame(y_pred).to_csv("hw08_target{}_test_predictions.csv".format(i))
    print("AUROC_TRAIN: {}".format(roc_auc_score(y_train, y_pred_train)))

def doForTarget(x,x_test,y,i):
    X_train,X_test, y_train = preprocessData(x,x_test, y,i)

    for m in models:
        applyModel(m, X_train, y_train, X_test, i)

def read(i):
    x_target = x[y['TARGET_{}'.format(i)].isna() == False]
    y_target = y[y['TARGET_{}'.format(i)].isna() == False]['TARGET_1']
    return x_target, y_target


def solve(i, x_target, y_target):
    print("TARGET{}:".format(i))
    print("--------")
    doForTarget(x_target, x_test, y_target, i)


models=[RandomForestClassifier(n_estimators=100, random_state=16)]

x=pd.read_csv("hw08_training_data.csv", index_col=0)
y=pd.read_csv("hw08_training_label.csv", index_col=0)
x_test=pd.read_csv("hw08_test_data.csv", index_col=0)

for i in range(6):
    x_target, y_target = read(i+1)
    solve(i+1, x_target, y_target)
