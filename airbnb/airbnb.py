# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# set random seed
np.random.seed(13)

# load data
df_train = pd.read_csv('/Users/ianmurray/Documents/kaggle/airbnb/input/train_users.csv')
df_test = pd.read_csv('/Users/ianmurray/Documents/kaggle/airbnb/input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

# load training and test data into a pandas dataframe
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# drop id and date_first_booking columns
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)

# replace/fill NaN/null fields
df_all = df_all.fillna(-1)

# process feature date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

# process feature timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# process feature age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

# process ohe features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

# split df into test and training data
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

# use xgboost XGBClassifier 
xgb = XGBClassifier(max_depth=8, learning_rate=0.075, n_estimators=250,
                    objective='multi:softprob', subsample=0.75, colsample_bytree=0.85, seed=13)                  
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  

# select the 5 highest probability classes
ids = []  # list ids
cts = []  # list countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# generate output 'pysub.csv'
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('/Users/ianmurray/Documents/kaggle/airbnb/output/pysub.csv',index=False)
