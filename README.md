# HousePricePrediction
I've used Machine Learning to create a model that predicts house prices in Melbourne :).

Libraries used include:

- Pandas
- Sklearn.

In further details, I have imported **ensemble.RandomForestRegressor**, **metrics.mean_absolute_error** and **model_selection.train_test_split** from **sklearn**


## Which Tree Model to use? 
To minimize Mean Absolute Error (MAE), I chose **RandomForestRegressor** over **DecisionTreeRegressor** with various max_leaf_nodes 5, 50, 500, 1000. The result appeared to produce a minimum MAE between 50 and 500. I chose 250 as my nodes. 
