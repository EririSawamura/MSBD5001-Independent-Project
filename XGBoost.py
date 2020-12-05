import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("train.csv", dayfirst=True, parse_dates=['date'])
dft = pd.read_csv("test.csv", dayfirst=True, parse_dates=['date'])

#Add year, month, day, hour
df['year'] = pd.Series([date.year for date in df['date']])
df['month'] = pd.Series([date.month for date in df['date']])
df['day'] = pd.Series([date.day for date in df['date']])
df['hour'] = pd.Series([date.hour for date in df['date']])
df['ymd'] = df['date'].dt.date

dft['year'] = pd.Series([date.year for date in dft['date']])
dft['month'] = pd.Series([date.month for date in dft['date']])
dft['day'] = pd.Series([date.day for date in dft['date']])
dft['hour'] = pd.Series([date.hour for date in dft['date']])
dft['ymd'] = dft['date'].dt.date

#Add weekday
week_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
df["day_of_week"] = df["date"].dt.day_name()
dft["day_of_week"] = dft["date"].dt.day_name()
for day in week_day:
    df[day] = 0
    dft[day] = 0
    df.loc[df["day_of_week"] == day, day] = 1
    dft.loc[dft["day_of_week"] == day, day] = 1
df.drop(["day_of_week", "date"], axis=1, inplace=True)
dft.drop(["day_of_week", "date"], axis=1, inplace=True)

#Add weather information
df_w = pd.read_csv('hongkong.csv', dayfirst=True, parse_dates=['date_time'])
df_w.set_index(['date_time'], inplace=True)
df.set_index('ymd', inplace=True)
df = df.join(df_w)
dft.set_index('ymd', inplace=True)
dft = dft.join(df_w)

#Add holiday
holidays = pd.to_datetime([ '2017-01-02', '2017-1-28', '2017-1-30', '2017-1-31',
                            '2017-4-4', '2017-4-5', '2017-4-15', '2017-4-17',
                            '2017-5-1', '2017-5-3', '2017-5-30', '2017-7-1',
                            '2017-10-2', '2017-10-5', '2017-10-28', '2017-12-25', '2017-12-26',
                            '2018-01-01', '2018-2-16', '2018-2-17', '2018-2-19',
                            '2018-3-30', '2018-3-31', '2018-4-2', '2018-4-5',
                            '2018-5-1', '2018-5-22', '2018-6-18', '2018-7-2',
                            '2018-9-25', '2018-10-1', '2018-10-17', '2018-12-25', '2018-12-26'])

holidays = pd.to_datetime(holidays)
holidays = pd.Series(1, index=holidays, name='holiday')
df = df.join(holidays)
df['holiday'].fillna(0, inplace=True)
dft = dft.join(holidays)
dft['holiday'].fillna(0,inplace=True)

#Model training
feature_list = ['year', 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','month', 'day',
                'hour', 'holiday', 'tempC', 'visibility', 'winddirDegree', 'windspeedKmph','humidity','cloudcover', 'WindChillC']
x = df.loc[: , feature_list]
test = dft.loc[:, feature_list]
y = df.loc[:, 'speed']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=1)

model = xgb.XGBRegressor(max_depth=7, learning_rate=0.09, n_estimators=450, objective='reg:squarederror')
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('MSE: ', mean_squared_error(y_test, pred))

dft['speed'] = model.predict(test)
res = dft[['id', 'speed']].set_index('id')
res.to_csv('result.csv')