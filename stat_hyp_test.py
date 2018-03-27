from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

class HypothesisTesting:
    
    
    def __init__(self, base_model_A, X, y, kFolds=5):
        self.ctrl_reg = base_model_A
        self.kFolds = kFolds
        self.control = None
        self.X = X
        self.y = y
        self.buildBaseModel(self.X, self.y)

    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
    
    
    # A sampling based bakeoff using *K-fold cross-validation*: 
    # it randomly splits the training set into K distinct subsets (k=30)
    # this bakeoff framework can be used for regression or classification
    # Control system is a linear regression based pipeline
    
    def buildBaseModel(self, X, y,):
        lin_scores = cross_val_score(self.ctrl_reg, X, y,
                                     scoring="neg_mean_squared_error", cv=self.kFolds, n_jobs=-1)
        self.control = lin_rmse_scores = np.sqrt(-lin_scores)
        self.display_scores(lin_rmse_scores)
        print('Control/Base Model is built')
   
    def evaluate(self, algo):     
       
        # Treatment system is a Decision Tree regression based pipeline
        scores = cross_val_score(algo, self.X, self.y,
                                 scoring="neg_mean_squared_error", cv=self.kFolds)
        treatment = tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)
        
        result_str = None
        # paired t-test; two-tailed p-value (aka two-sided)
        (t_score, p_value) = stats.ttest_rel(self.control, treatment)
        result_str = str("The p-value is %0.5f for a t-score of %0.5f." % (p_value, t_score))
        # "The p-value is 0.00019 for a t-score of -4.28218." 
        if p_value > 0.05 / 2:  # Two sided 
            result_str = str('There is no significant difference between the two machine learning pipelines (Accept H0)')
        else:
            print('The two machine learning pipelines are different (reject H0) \n(t_score, p_value) = (%.2f, %.5f)' % (t_score, p_value))
            if t_score < 0.0:
                result_str = str('Machine learning pipeline A is better than B')
            else:
                result_str = str('Machine learning pipeline B is better than A')
        return p_value, result_str
