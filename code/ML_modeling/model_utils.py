import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def pre_process_for_ML(df): 
    X = df.drop(["name", "position", "team", "5_gw_fpl_pts"], axis=1)
    y = df["5_gw_fpl_pts"] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    return X_train, X_test, y_train, y_test


def eval_model(model, X_test): 
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'rmse: {rmse:.2f}')
    print(f'R2 score: {r2:.2f}')


def tune_model(df, num_trials): 
    X = df.drop(["name", "position", "team", "5_gw_fpl_pts"], axis=1)
    y = df["5_gw_fpl_pts"] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    def objective(trial):
        # Define hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }
        
        # Create and train XGBoost model
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on validation set and calculate MSE
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred) 
        return r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    return study.best_trial


def get_feature_importance(model, X_train): 
    importances = model.feature_importances_

    # Get corresponding feature names
    feature_names = X_train.columns

    # Sort indices of feature importance in descending order
    indices = np.argsort(importances)[::-1]

    # Print feature importances
    for i, index in enumerate(indices):
        print(f"{i + 1}. {feature_names[index]}: {importances[index]}")

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90, fontsize=10)
    plt.subplots_adjust(bottom=0.3)
    plt.show()