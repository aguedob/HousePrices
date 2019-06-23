#%%
# Code you have previously used to load data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
#from learntools.core import *


def prepareData(df):
    df=df.drop(["MiscFeature","Fence","PoolQC","Alley","GarageCond","FireplaceQu","GarageFinish","BsmtQual","GarageQual"],axis=1)
    df["LotFrontage"]=df["LotFrontage"].fillna(0)
    df["MasVnrType"]=df["MasVnrType"].fillna("None")
    df["GarageType"]=df["GarageType"].fillna("None")
    df["Electrical"]=df["Electrical"].fillna("SBrkr")
    df.GarageYrBlt.fillna(df.YearBuilt,inplace=True)
    df["BsmtCond"]=df["BsmtCond"].fillna("TA")
    df["BsmtFinType2"]=df["BsmtFinType2"].fillna("Unf")
    df["BsmtFinType1"]=df["BsmtFinType1"].fillna("Unf")
    df["BsmtExposure"]=df["BsmtExposure"].fillna("No")
    df["MSZoning"]=df["MSZoning"].fillna("RF")


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


    # Categorical boolean mask
    #categorical_feature_mask = df.dtypes==object
    # filter categorical columns using mask and turn it into a list
    #categorical_cols = df.columns[categorical_feature_mask].tolist()
    #le = LabelEncoder()
    # apply le on categorical feature columns
    #df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    #ohe = OneHotEncoder()
    # instantiate OneHotEncoder
    #ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 
    # categorical_features = boolean mask for categorical columns
    # sparse = False output an array not sparse matrix

    # apply OneHotEncoder on categorical feature columns
    #X_ohe = ohe.fit_transform(df) # It returns an numpy array
    return X



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Train set size:", train_data.shape)
print("Test set size:", test_data.shape)

# Deleting detected outliers
train_data = train_data[train_data.GrLivArea < 4500]
# Reset index
train_data.reset_index(drop=True, inplace=True)

train_features = train_data.drop(['SalePrice'], axis=1)
test_features = test_data


#%%
# Create target object and call it y
y = train_data.SalePrice

features = train_data.columns[:-1]
# Prepare X
X = prepareData(train_data[features])

input_X = train_data[features]
test_X = test_data[features]

combined_X = pd.concat([input_X,test_X])

combined_X = prepareData(combined_X)
print(combined_X.shape)


input_X = combined_X.iloc[:len(y), :]
test_X = combined_X.iloc[len(input_X):, :]

# %%
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(input_X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1,n_estimators=100)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


xgb_model = XGBRegressor(random_state=1, objective='reg:squarederror', silent=1, n_estimators=2800, learning_rate=0.015, max_depth=4, subsample=0.80, colsample_bytree=0.7, gamma=1  )
xgb_model.fit(train_X, train_y)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)
print("Validation MAE for XGB Regressor Model: {:,.0f}".format(xgb_val_mae))






# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = XGBRegressor(random_state=1, objective='reg:squarederror', silent=1, n_estimators=2800, learning_rate=0.015, max_depth=4, subsample=0.80, colsample_bytree=0.7, gamma=1 )

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(input_X,y)



#%%
# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



#%%




