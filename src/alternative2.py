#%%
# Code you have previously used to load data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing
from xgboost import XGBRegressor

def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            print(df)


def prepareData(df):
    df=df.drop(["MiscFeature","Fence","PoolQC","Alley","GarageCond","FireplaceQu","GarageFinish","BsmtQual","GarageQual","Id"],axis=1)    
    df['MSSubClass']=df['MSSubClass'].astype(str)
    df['MiscVal']=df['MiscVal'].astype(str)
    df['MoSold']=df['MoSold'].astype(str)
    df['YrSold']=df['YrSold'].astype(str)
    df["LotFrontage"]=df["LotFrontage"].fillna(0)
    df["MasVnrType"]=df["MasVnrType"].fillna("None")
    df["GarageYrBlt"].fillna(df["YearBuilt"],inplace=True)
    df['KitchenQual'].fillna(df['OverallQual'],inplace=True)
    df["GarageType"]=df["GarageType"].fillna("None")
    df["SaleCondition"]=df["SaleCondition"].fillna(df["SaleCondition"].mode()[0])
    df["Electrical"]=df["Electrical"].fillna(df["Electrical"].mode()[0])    
    df["Functional"]=df["Functional"].fillna("Typ")    
    df["BsmtCond"]=df["BsmtCond"].fillna("TA")
    df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0], inplace=True)
    df["BsmtFinType1"].fillna(df["BsmtFinType1"].mode()[0], inplace=True)
    df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0], inplace=True)
    df["MSZoning"].fillna("RF", inplace=True)
    
    pd.option_context('display.max_rows', None)

    objects = []
    for i in df.columns:
        if df[i].dtype == object:
            objects.append(i)

    df.update(df[objects].fillna('None'))

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numbers = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numbers.append(i)
    df.update(df[numbers].fillna(0))

    X = pd.get_dummies(df, prefix_sep='_', drop_first=True)
    return X



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Train set size:", train_data.shape)
print("Test set size:", test_data.shape)

# Deleting detected outlier
train_data = train_data[train_data.GrLivArea < 4500]

# Reset index
train_data.reset_index(drop=True, inplace=True)

train_features = train_data.drop(['SalePrice'], axis=1)
test_features = test_data


#%%
# Create target object and call it y (normalize with log(f+1) )
y = np.log(train_data.SalePrice+1)

features = train_data.columns[:-1]
# Prepare X
X = prepareData(train_data[features])

input_X = train_data[features]
test_X = test_data[features]

combined_X = pd.concat([input_X,test_X])

combined_X = prepareData(combined_X)
print("Combined set size:", combined_X.shape)


input_X = combined_X.iloc[:len(y), :]
test_X = combined_X.iloc[len(input_X):, :]

#%%

# Fitting with all data

gbr = GradientBoostingRegressor(max_depth=4, n_estimators=150, random_state=1)
gbr.fit(input_X, y)
gbr_score = np.sqrt(-cross_val_score(gbr, input_X, y, cv=5, scoring="neg_mean_squared_error")).mean()
print("Validation GBR : ", gbr_score)

xgbr = XGBRegressor(random_state=1, objective='reg:squarederror', max_depth=5, n_estimators=400)
xgbr.fit(input_X, y)
xgbr_val = np.sqrt(-cross_val_score(xgbr, input_X, y, cv=5, scoring="neg_mean_squared_error")).mean()
print("Validation XGBR : ", xgbr_val)

lsr = Lasso(alpha=0.00047,random_state=1)
lsr.fit(input_X, y)
lasso_val = np.sqrt(-cross_val_score(lsr, input_X, y, cv=5, scoring="neg_mean_squared_error")).mean()
print("Validation Lasso : ", lasso_val)

rr = Ridge(alpha=13, random_state=1)
rr.fit(input_X, y)
ridge_val = np.sqrt(-cross_val_score(rr, input_X, y, cv=5, scoring="neg_mean_squared_error")).mean()
print("Validation Ridge : ", ridge_val)

print("Combined : ", 0.1 * gbr_score + 0.3 * xgbr_val +  0.3 * lasso_val + 0.3 * ridge_val)

# Get train predict for manual adjustment
train_predict = 0.1 * gbr.predict(input_X) + 0.3 * xgbr.predict(input_X) + 0.3 * lsr.predict(input_X) + 0.3 * rr.predict(input_X)

# Manual adjustment
q1 = pd.DataFrame(train_predict).quantile(0.0092) # << Manually calculated
pre_df = pd.DataFrame(train_predict)
pre_df["SalePrice"] = train_predict
pre_df = pre_df[["SalePrice"]]
pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.99
train_predict = np.array(pre_df.SalePrice)
plt.figure(figsize=(8, 8))
plt.scatter(y, train_predict)
plt.plot(range(10, 15), range(10, 15), color="blue")



# make predictions which we will submit. 
test_predict = 0.1 * gbr.predict(test_X) + 0.3 * xgbr.predict(test_X) + 0.3 * lsr.predict(test_X) + 0.3 * rr.predict(test_X)
q1 = pd.DataFrame(test_predict).quantile(0.0092)
pre_df = pd.DataFrame(test_predict)
pre_df["SalePrice"] = test_predict
pre_df = pre_df[["SalePrice"]]
pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.96
test_predict = np.array(pre_df.SalePrice)

# The lines below shows how to save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': np.exp(test_predict)-1})
output.to_csv('submission2.csv', index=False)
