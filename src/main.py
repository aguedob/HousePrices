# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from learntools.core import *
import os
cwd = os.getcwd()
print(cwd)

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = './input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', 'BsmtFinSF1', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',  'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'Neighborhood','GrLivArea','TotalBsmtSF']
X = home_data[features]


neighborhoods = X.Neighborhood.unique()
neighborhood_dict = dict(zip(neighborhoods, range(len(neighborhoods))))

X=X.applymap(lambda s: neighborhood_dict.get(s) if s in neighborhood_dict else s)

initial_X = X
initial_y = y


# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
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





















def prepareDataFrame(path):
    # load data
    home_data =  pd.read_csv(path)
    
    # map to integers all columns
    for y in home_data.columns:
        if(home_data[y].dtype == np.float64 or home_data[y].dtype == np.int64):
            pass
        else:
            neighborhoods = home_data[y].unique()
            neighborhood_dict = dict(zip(neighborhoods, range(len(neighborhoods))))
            home_data=home_data.applymap(lambda s: neighborhood_dict.get(s) if s in neighborhood_dict else s)
    home_data=home_data.fillna(0)
    return home_data




# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1, n_estimators=100)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(initial_X,initial_y)




# path to file you will use for predictions
test_data_path = './input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path).fillna(0)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

neighborhoods = test_X.Neighborhood.unique()
neighborhood_dict = dict(zip(neighborhoods, range(len(neighborhoods))))

test_X=test_X.applymap(lambda s: neighborhood_dict.get(s) if s in neighborhood_dict else s)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)







