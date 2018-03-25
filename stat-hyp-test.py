from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# A sampling based bakeoff using *K-fold cross-validation*: 
# it randomly splits the training set into K distinct subsets (k=30)
# this bakeoff framework can be used for regression or classification
# Control system is a linear regression based pipeline
X = None;y = None;A = None;B = None


def setData(X, y):
    global X, y
    X = X;
    y = y;


ctrl_reg = None;treat_reg = None


def setAlgos(ctrl_reg, treat_reg):
    global ctrl_reg, treat_reg
    ctrl_reg = ctrl_reg
    treat_reg = treat_reg


kFolds = 30
lin_scores = cross_val_score(ctrl_reg, X, y,
                             scoring="neg_mean_squared_error", cv=kFolds)
control = lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Treatment system is a Decision Tree regression based pipeline
scores = cross_val_score(treat_reg, X, y,
                         scoring="neg_mean_squared_error", cv=kFolds)
treatment = tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

# paired t-test; two-tailed p-value (aka two-sided)
(t_score, p_value) = stats.ttest_rel(control, treatment)
print("The p-value is %0.5f for a t-score of %0.5f." % (p_value, t_score))
# "The p-value is 0.00019 for a t-score of -4.28218." 
if pvalue > 0.05 / 2:  # Two sided 
    print('There is no significant difference between the two machine learning pipelines (Accept H0)')
else:
    print('The two machine learning pipelines are different (reject H0) \n(t_score, p_value) = (%.2f, %.5f)' % (t_score, p_value))
    if t_score < 0.0:
        print('Machine learning pipeline A is better than B')
    else:
        print('Machine learning pipeline B is better than A')
