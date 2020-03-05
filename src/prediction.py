import pickle

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Helper function that makes prediction from user inputs

def binary_predictor(pickle_model, user_array):
    # model = pickle.load( open(pickle_model, 'rb'))

    soft_prediction = pickle_model.predict_proba(user_array)
    if soft_prediction[0][1] > 0.4:
        return 1    # 1 = Yes
    else: return 0    # 0 = No
    # return soft_prediction, user_array
