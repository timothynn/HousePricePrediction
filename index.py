from statistics import mean
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('train.csv')
y = home_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X =home_data[features]

print(X.head())



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
dt_val_predictions = dt_model.predict(val_X)
dt_val_mae = mean_absolute_error(dt_val_predictions, val_y)

print("Validation MAE for Decision Tree Model: {:,.0f}".format(dt_val_mae))


