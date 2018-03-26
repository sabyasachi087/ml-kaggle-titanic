import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from CombinedAttributeHeader import CombinedAttributesAdder
from DataFrameSelector import DataFrameSelector
from CategoricalEncoder import CategoricalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from stat_hyp_test import HypothesisTesting
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import time

HOUSING_PATH = 'datasets/housing/'
from sklearn.model_selection import train_test_split

def mean_absolute_percentage_error(y_true, y_pred, **kwargs): 
    """
    Use of this metric is not recommended because can cause division by zero
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """
    return np.mean(np.abs((y_true.ravel() - y_pred.ravel()) / y_true.ravel())) * 100

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)



from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = None;strat_test_set = None
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

housing_test = strat_test_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_test_labels = strat_test_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)
housing_test_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),  # derive new features 
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_test = num_pipeline.fit_transform(housing_test_num)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
    ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_test_prepared = full_pipeline.fit_transform(housing_test)

model_maps = dict()
model_maps["Linear_Regression"] = LinearRegression()
model_maps["Logistic_Regression"] = LogisticRegression()
model_maps["DecisionTreeRegressor"] = DecisionTreeRegressor()
model_maps["RandomForestRegressor"] = RandomForestRegressor()
model_maps["SupportVectorRegressor"] = SVR()

results = pd.DataFrame(columns=["Hardware", "ExpID", "RMSETrainCF", "RMSETest", "MAPETrainCF", "MAPETest", "p-value", "TrainTime(s)", "TestTime(s)", "Experiment description"])

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

for name, algo in model_maps.items():
    indx = len(results)
    results.loc[indx] = ["Corei3/8GB", indx + 1, 0, 0, 0, 0, 0, 0, 0, "Verifying " + str(name)]
    start_time = time.time()
    algo.fit(housing_prepared, housing_labels)
    results.loc[indx]["TrainTime"] = time.time() - start_time
    scores = cross_val_score(algo, housing_prepared, housing_labels,
                         scoring=mape_scorer, cv=5)
    results.loc[indx]["MAPETrainCF"] = scores.mean()
    scores = cross_val_score(algo, housing_prepared, housing_labels,
                         scoring=mean_squared_error, cv=5)
    results.loc[indx]["RMSETrainCF"] = np.sqrt(scores).mean()
    # Test / Prediction
    start_time = time.time()
    housing_predictions = algo.predict(housing_test_prepared)
    results.loc[indx]["TestTime"] = time.time() - start_time
    algo_rmse = np.sqrt(mean_squared_error(housing_test_labels, housing_predictions))
    results.loc[indx]["RMSETest"] = algo_rmse.mean()
    algo_mape = np.array(mape_scorer(housing_test_labels, housing_predictions))
    results.loc[indx]["MAPETest"] = algo_mape.mean()
    
