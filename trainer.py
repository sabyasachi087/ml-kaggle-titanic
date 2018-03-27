import curator as crt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from DataFrameSelector import DataFrameSelector
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import FeatureUnion
from CategoricalEncoder import CategoricalEncoder

DATA_FILE_PATH = 'data/train.csv'
TEST_FILE_PATH = 'data/test.csv'
titanic_data = crt.preProcessAndGet(DATA_FILE_PATH)
titanic_label = titanic_data["Survived"]
titanic_data.drop("Survived", axis=1, inplace=True)
titanic_passenger_ids = titanic_data["PassengerId"]
titanic_data.drop("PassengerId", axis=1, inplace=True)

cat_attribs = ["Embarked", "Sex", "title"]
titanic_data_num = titanic_data.copy()
for cat_attrb in cat_attribs:
    titanic_data_num = titanic_data_num.drop(cat_attrb, axis=1)
num_attribs = titanic_data_num.columns.values.tolist()
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
    ])
    
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# titanic_data_preapred = full_pipeline.fit_transform(titanic_data)

# test_titanic_data = crt.preProcessAndGet(TEST_FILE_PATH)
# test_titanic_passenger_ids = test_titanic_data["PassengerId"]
# test_titanic_data.drop("PassengerId", axis=1, inplace=True)
# Run training program

# logReg = LinearRegression()
# logReg.fit(titanic_data_preapred, titanic_label)
# titanic_predictions = logReg.predict(titanic_data_preapred)
# log_mse = mean_squared_error(titanic_label, titanic_predictions)
# log_mse_rmse = np.sqrt(log_mse)
# print(log_mse)

test_titanic_data = crt.preProcessAndGet(TEST_FILE_PATH)
test_titanic_passenger_ids = test_titanic_data["PassengerId"]
test_titanic_data.drop("PassengerId", axis=1, inplace=True)

full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("logRegression", LinearRegression())
    ])

full_pipeline_with_predictor.fit(titanic_data, titanic_label)
predict_labels = pd.Series(full_pipeline_with_predictor.predict(test_titanic_data)).apply(lambda x : 1 if x >= 0.5 else 0)

result = pd.DataFrame({'PassengerId': test_titanic_passenger_ids, 'Survived':predict_labels})
result.to_csv('sabya_submission.csv', index=False)

# scores = cross_val_score(full_pipeline_with_predictor, titanic_data, titanic_label,
#                          scoring="neg_mean_squared_error", cv=20)
# log_rmse_scores = np.sqrt(-scores)
#   
#    
# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#   
#    
# display_scores(log_rmse_scores)
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42, n_estimators=30, max_features=4)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error')
grid_search.fit(full_pipeline.fit_transform(titanic_data), titanic_label)
print(grid_search.best_params_)
