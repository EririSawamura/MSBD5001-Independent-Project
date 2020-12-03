import xgboost as xgb
import pandas as pd

from xgboost import plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
pd.options.mode.chained_assignment = None

df = pd.read_csv("train.csv")
df["time"] = df["date"].apply(lambda x : x[-5:-3])
df["day"] = df["date"].apply(lambda x : x.split('/')[0])
df["month"] = df["date"].apply(lambda x : x.split('/')[1])
df["year"] = df["date"].apply(lambda x : x.split('/')[2][0:4])

#Add feature: Holiday
df['holiday'] = 0
df.loc[(df.month == '1') & (df.year == '2017') & (df.day.isin(['1', '2', '27', '28', '29', '30', '31'])), 'holiday'] = 1
df.loc[(df.month == '2') & (df.year == '2017') & (df.day.isin(['1', '2'])), 'holiday'] = 1
df.loc[(df.month == '4') & (df.year == '2017') & (df.day.isin(['2', '3', '4', '29', '30'])), 'holiday'] = 1
df.loc[(df.month == '5') & (df.year == '2017') & (df.day.isin(['1', '28', '29', '30'])), 'holiday'] = 1
df.loc[(df.month == '10') & (df.year == '2017') & (df.day.isin(['1', '2', '3', '4', '5', '6', '7', '8'])), 'holiday'] = 1
df.loc[(df.month == '12') & (df.year == '2017') & (df.day.isin(['31'])), 'holiday'] = 1
df.loc[(df.month == '1') & (df.year == '2018') & (df.day.isin(['1'])), 'holiday'] = 1
df.loc[(df.month == '2') & (df.year == '2018') & (df.day.isin(['15', '16', '17', '18', '19', '20', '21'])), 'holiday'] = 1
df.loc[(df.month == '4') & (df.year == '2018') & (df.day.isin(['5', '6', '7'])), 'holiday'] = 1
df.loc[(df.month == '6') & (df.year == '2018') & (df.day.isin(['16', '17', '18'])), 'holiday'] = 1
df.loc[(df.month == '9') & (df.year == '2018') & (df.day.isin(['24'])), 'holiday'] = 1
df.loc[(df.month == '10') & (df.year == '2018') & (df.day.isin(['1', '2', '3', '4', '5', '6', '7'])), 'holiday'] = 1

#Add feature: Season
df['spring'] = 0
df['summer'] = 0
df['autumn'] = 0
df['winter'] = 0
df['winter'].iloc[0:783] = 1
df['spring'].iloc[783:2967] = 1
df['summer'].iloc[2967:5222] = 1
df['autumn'].iloc[5222:7430] = 1
df['winter'].iloc[7430:9246] = 1
df['spring'].iloc[9246:10566] = 1
df['summer'].iloc[10566:11975] = 1
df['autumn'].iloc[11975:13260] = 1
df['winter'].iloc[13260:14006] = 1

#Add feature: Weekday
df["month"] = df["month"].apply(lambda x: '{0:0>2}'.format(x))
df["day"] = df["day"].apply(lambda x: '{0:0>2}'.format(x))
df["date"] = df[['year', 'month', 'day']].agg('-'.join, axis=1)
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.day_name()

week_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for day in week_day:
    df[day] = 0
    df.loc[df["day_of_week"] == day, day] = 1

df["time"] = df["time"].astype("float64")
df["day"] = df["day"].astype("float64")
df["month"] = df["month"].astype("float64")
df["year"] = df["year"].astype("float64")

ydata = df["speed"]
df.drop(["date", "id", "day_of_week", "speed"], axis = 1, inplace=True)

xdata = df

#Load test data
dft = pd.read_csv("test.csv")
dft["time"] = dft["date"].apply(lambda x : x[-5:-3])
dft["day"] = dft["date"].apply(lambda x : x.split('/')[0])
dft["month"] = dft["date"].apply(lambda x : x.split('/')[1])
dft["year"] = dft["date"].apply(lambda x : x.split('/')[2][0:4])
dayt = dft["day"]

#Same feature as training data
dft['holiday'] = 0
dft.loc[(dft.month == '1') & (dft.year == '2018') & (dft.day.isin(['1'])), 'holiday'] = 1
dft.loc[(dft.month == '2') & (dft.year == '2018') & (dft.day.isin(['15', '16', '17', '18', '19', '20', '21'])), 'holiday'] = 1
dft.loc[(dft.month == '4') & (dft.year == '2018') & (dft.day.isin(['5', '6', '7'])), 'holiday'] = 1
dft.loc[(dft.month == '6') & (dft.year == '2018') & (dft.day.isin(['16', '17', '18'])), 'holiday'] = 1
dft.loc[(dft.month == '9') & (dft.year == '2018') & (dft.day.isin(['24'])), 'holiday'] = 1
dft.loc[(dft.month == '10') & (dft.year == '2018') & (dft.day.isin(['1', '2', '3', '4', '5', '6', '7'])), 'holiday']

dft['spring'] = 0
dft['summer'] = 0
dft['autumn'] = 0
dft['winter'] = 0
dft['winter'].iloc[0:320] = 1
dft['spring'].iloc[320:1160] = 1
dft['summer'].iloc[1160:2007] = 1
dft['autumn'].iloc[2007:2930] = 1
dft['winter'].iloc[2930:3504] = 1

dft["month"] = dft["month"].apply(lambda x: '{0:0>2}'.format(x))
dft["day"] = dft["day"].apply(lambda x: '{0:0>2}'.format(x))
dft["date"] = dft[['year', 'month', 'day']].agg('-'.join, axis=1)
dft["date"] = pd.to_datetime(dft["date"])
dft["day_of_week"] = dft["date"].dt.day_name()

for day in week_day:
    dft[day] = 0
    dft.loc[dft["day_of_week"] == day, day] = 1

dft.drop(["date", "id", "day_of_week"], axis = 1, inplace=True)
dft["time"] = dft["time"].astype("float64")
dft["day"] = dft["day"].astype("float64")
dft["month"] = dft["month"].astype("float64")
dft["year"] = dft["year"].astype("float64")
xtest = dft

#Split data for training and validation
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=1)

#Hyperparameter list
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0,
    'max_depth': 40,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'min_child_weight': 2, 
    'eta': 0.01,
    'seed': 1000,
    'nthread': 4,
    'reg_alpha': 2,
}

dtrain = xgb.DMatrix(x_train, y_train)
num_rounds = 850
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)
predict_y_train = model.predict(xgb.DMatrix(x_train))
predict_y_test = model.predict(xgb.DMatrix(x_test))
trainMSE = metrics.mean_squared_error(y_train, predict_y_train)
testMSE = metrics.mean_squared_error(y_test, predict_y_test)
print("trainMSE:", trainMSE)
print("testMSE:", testMSE)

dtrain = xgb.DMatrix(xdata, ydata)
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

res = model.predict(xgb.DMatrix(xtest))
sub = pd.read_csv("sampleSubmission.csv")
del sub['speed']
sub['speed'] = res
sub.to_csv("result5.csv", index=False)


