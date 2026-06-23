import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error,
    explained_variance_score
)

def train_model(df: pd.DataFrame, target: str):

    # Remove outliers from all numeric columns
    z_scores = np.abs(stats.zscore(df.select_dtypes(include='number')))
    df = df[(z_scores < 3).all(axis=1)]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    import os
    os.makedirs("model", exist_ok=True)
    with open("model/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    X_test  = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(random_state=42)

    # Reduced param grid
    param_grid = {
       'n_estimators'      : [100, 300, 500],
        'max_depth'         : [10, 20, None],
        'min_samples_split' : [2, 5],
        'max_features'      : ['sqrt', 'log2']
        }

    grid = GridSearchCV(
        estimator  = model,
        param_grid = param_grid,
        cv         = 5,
        scoring    = 'neg_mean_squared_error',
        n_jobs     = -1,
        verbose    = 2
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print("Best Params:", grid.best_params_)

    return model,X_test, y_test


def metrix_cal(y_test, y_pred):
    mae      = mean_absolute_error(y_test, y_pred)
    mse      = mean_squared_error(y_test, y_pred)
    r2       = r2_score(y_test, y_pred)
    mape     = mean_absolute_percentage_error(y_test, y_pred)
    evs      = explained_variance_score(y_test, y_pred)
    return mae, mse, r2, mape, evs


def save_model(model: object, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


def load_model(file_path: str) -> object:
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


def inferance_model(model: object, X_test):
    y_pred = model.predict(X_test)
    return y_pred