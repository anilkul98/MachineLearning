import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings(action="ignore")
def isSpecialDate(date):
    specialDays=[[2018,1,1],[2018,4,23],[2018,5,1],[2018,5,19],[2018,6,14],[2018,6,15],[2018,6,16],
                 [2018,6,17], [2018,7,15],[2018,8,20],[2018,8,21],[2018,8,22],[2018,8,23],[2018,8,24],
                 [2018,8,30],[2018,10,29],[2019, 1, 1], [2019, 4, 23], [2019, 5, 1],[2019, 5, 19],
                 [2019, 6, 3], [2019, 6, 4], [2019, 6, 5],[2019, 6, 6], [2019, 7, 15],[2019, 8, 10],
                 [2019, 8, 11], [2019, 8, 12],[2019, 8, 13], [2019, 8, 14],[2019, 8, 30], [2019, 10, 29]]
    if specialDays.__contains__(date): return 1
    else: return 0

def isBeforeSpecialDate(date):
    specialDays=[[2018,12,31],[2018,12,30],[2018,4,21],[2018,4,22],[2018,4,30],[2018,4,29],[2018,5,18],
                 [2018, 5, 17],[2018,6,12],[2018,6,13],[2018,7,14],[2018,7,13],[2018,8,16],[2018,8,17],
                 [2018,8,18],[2018,8,19],[2018,8,20],[2018,8,28],[2018,8,29],[2018,10,28],[2018,10,27],
                 [2019,12,31],[2019,12,30],[2019,4,21],[2019,4,22],[2019,4,30],[2019,4,29],[2019,5,18],
                 [2019, 5, 17],[2019, 5, 30],[2019, 5, 31],[2019, 6, 1], [2019, 6, 2], [2019, 7, 14],
                 [2019, 7, 13],[2019, 8, 6],[2019, 8, 7],[2019, 8, 8], [2019, 8, 9], [2019, 8, 28],
                 [2019, 8, 29], [2019, 10, 28], [2019, 10, 29]]
    if specialDays.__contains__(date): return 1
    else: return 0

def preprocessData(x):
    x.insert(0, 'BEFORE_SPECIAL_DATES',
             x.apply(lambda row: isBeforeSpecialDate([row['YEAR'], row['MONTH'], row['DAY']]), axis=1))
    x.insert(0, 'SPECIAL_DATES', x.apply(lambda row: isSpecialDate([row['YEAR'], row['MONTH'], row['DAY']]), axis=1))
    x.insert(0, 'WEEKEND',
             x.apply(lambda row: 0 if (pd.Timestamp(row['YEAR'], row['MONTH'], row['DAY']).weekday() <= 5) else 1,
                     axis=1))
    x.insert(0, 'SALARY_DAYS', x.apply(
        lambda row: 1 if (row['DAY'] <= 5 or (row['DAY'] >= 25 and row['DAY'] <= 31)) and row['WEEKEND'] != 1 else 0,
        axis=1))
    x = pd.concat([x, pd.get_dummies(x['TRX_TYPE'], prefix='TRX_TYPE')], axis=1)
    x.drop(['TRX_TYPE'], axis=1, inplace=True)
    x = pd.concat([x, pd.get_dummies(x['IDENTITY'], prefix='IDENTITY')], axis=1)
    x.drop(['IDENTITY'], axis=1, inplace=True)
    x = pd.concat([x, pd.get_dummies(x['REGION'], prefix='REGION')], axis=1)
    x.drop(['REGION'], axis=1, inplace=True)
    return x

X_train=pd.read_csv("training_data.csv")
X_train=pd.DataFrame(X_train)
X_test=pd.read_csv("test_data.csv")
X_test=pd.DataFrame(X_test)
X_train=preprocessData(X_train)
X_test=preprocessData(X_test)

Y_train = X_train['TRX_COUNT']
X_train.drop(['TRX_COUNT'], axis=1, inplace=True)

model = RandomForestRegressor(n_estimators=500, random_state=16)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv("test_predictions.csv",header=0, index=0)