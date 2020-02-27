from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def logreg(X, y, C, max_iter):
    # Split data into training set and testing set
    # By default, 75% of the data set is used to for training and 
    # 25% of the data is used to test the model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Instantiate Logistic Regression model and fit it to scaled data
    lr = LogisticRegression(solver='lbfgs', C=C, max_iter=max_iter)
    lr.fit(X_train, y_train)
    
    soft_yes = lr.predict_proba(X_test)
    hard_yes = lr.predict(X_test)
    
    # logloss and others requires the probabilities that Yes or 1 is predicted
    logl = metrics.log_loss(y_test, soft_yes)  
    fpr, tpr, _ = metrics.roc_curve(y_test, soft_yes[:, 1])
    auc = metrics.roc_auc_score(y_test, soft_yes[:, 1])
    
    # Precision and accuracy requires y-predictions as (0, 1)
    accuracy = metrics.accuracy_score(y_test, hard_yes)
    precision = metrics.precision_score(y_test, hard_yes)
    recall = metrics.recall_score(y_test, hard_yes)

    parameters, intercept = lr.coef_, lr.intercept_
    metrics_str = f'Logistic Regression:  Accuracy: {accuracy:.4f}.  Precision: {precision:.4f}.  Recall: {recall:.4f}.  Log-loss: {logl:.4f}.  AUC: {auc:.4f}'

    return metrics_str, {'Model': 'Log. Regression',
                        'X_test': X_test,    # For plotting at the end
                        'y_test': y_test,    # For plotting at the end
                        'hard_predictions': hard_yes,
                        'prediction probs': soft_yes,
                # For logistic regression, feature importances can be extracted from beta_coefficients
                        'beta_coeffs': parameters,    
                        'intercept': intercept,
                        'false pos rate': fpr,
                        'true pos rate': tpr,
                # This returns parameters used in function call
                        'parameters': lr.get_params()    
                        }
    


def ensemble(model_name, X, y, n_estimators, max_depth, min_samples_split, min_samples_leaf, learning_rate=None):
    # Do not need to scale the data for ensemble methods
    # By default, 75% of the data set is used to for training and 
    # 25% of the data is used to test the model
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    if model_name == 'RandomForest':
        # Instantiate RandomForestClassifier and fit
        mod = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, bootstrap=True)

    if model_name == "GradientBoost":
        # Instantiate GradientBoostingClassifier and fit
        mod = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, 
                                        learning_rate = learning_rate, subsample=0.5)
    
    mod.fit(X_train, y_train)        

    soft_yes = mod.predict_proba(X_test)
    hard_yes = mod.predict(X_test)
    
    # logloss and others requires the probabilities that Yes or 1 is predicted
    logl = metrics.log_loss(y_test, soft_yes)  
    fpr, tpr, _ = metrics.roc_curve(y_test, soft_yes[:, 1])
    auc = metrics.roc_auc_score(y_test, soft_yes[:, 1])
    
    # Precision and accuracy requires y-predictions as (0, 1)
    accuracy = metrics.accuracy_score(y_test, hard_yes)
    precision = metrics.precision_score(y_test, hard_yes)
    recall = metrics.recall_score(y_test, hard_yes)

    metrics_str = f'{model_name}:  Accuracy: {accuracy:.4f}.  Precision: {precision:.4f}.  Recall: {recall:.4f}.  Log-loss: {logl:.4f}.  AUC: {auc:.4f}'

    return metrics_str, {'Model': model_name,
                        'X_test': X_test,    # For plotting at the end
                        'y_test': y_test,    # For plotting at the end
                        'hard_predictions': hard_yes,
                        'prediction probs': soft_yes,
                        'feature importances': mod.feature_importances_,
                        'false pos rate': fpr,
                        'true pos rate': tpr,
                # This returns parameters used in function call
                        'parameters': mod.get_params()    
                        }



# if __name__ == "__main__":
#     from data_cleaning import load_training_data
#     data = load_training_data()
#     X = data.drop(columns='fraud')
#     y = data['fraud']

#     logreg(X, y)
#     ensemble(X, y)
