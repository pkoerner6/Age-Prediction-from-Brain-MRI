

import pandas as pd
import numpy as np
from typing import List, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
import lightgbm as lgbm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import structlog
log = structlog.get_logger()


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  df_x = pd.read_csv("X_train.csv")
  df_y = pd.read_csv("y_train.csv")
  df_y.drop(columns=['id'], inplace=True)
  df_x_test = pd.read_csv("X_test.csv")
  return df_x, df_y, df_x_test


# imputation of missing values
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
  for col in df.columns:
    mean = df[col].mean()
    df[col].fillna(mean, inplace=True)
  return df


# outlier detection
def remove_outliers(df: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:

  for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify and remove outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    df_clean = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

  outliers_indx = outliers.index
  df_y = df_y.drop(index=outliers_indx)
  df_y.reset_index(drop=True, inplace=True)
  
  df_clean.reset_index(drop=True, inplace=True)
  assert len(df_y) == len(df_clean)
  return df_clean, df_y


# feature selection
def feature_selection_with_elbow_method(df_x: pd.DataFrame, df_y: pd.DataFrame, df_x_test: pd.DataFrame) -> None:
    max_features = 300  # maximum number of features to consider
    num_features_to_test = min(max_features, df_x.shape[1])

    r2_scores = []

    for k in tqdm(range(100, num_features_to_test + 1, 20)):
        best_features = SelectKBest(score_func=mutual_info_classif, k=k)
        best_features.fit(df_x, df_y)

        selected_features = best_features.get_support(indices=True)
        df_x_selected = df_x.iloc[:, selected_features]

        x_train, x_test, y_train, y_test = train_test_split(df_x_selected, df_y['y'], test_size=0.2, random_state=42)
        regressor = ExtraTreesRegressor(n_estimators=100, random_state=42)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    plt.plot(range(100, num_features_to_test + 1, 20), r2_scores, marker='o')
    plt.title('Elbow Method for Feature Selection')
    plt.xlabel('Number of Features')
    plt.ylabel('R-squared Score')
    plt.show()
    return

def feature_selection(df_x: pd.DataFrame, df_y: pd.DataFrame, df_x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  k = 140 
  best_features = SelectKBest(score_func=mutual_info_classif, k=k)
  best_features.fit(df_x, df_y)

  feature_scores = pd.DataFrame({'Feature': df_x.columns, 'Score': best_features.scores_, 'P-Value': best_features.pvalues_})
  feature_scores = feature_scores.sort_values(by='Score', ascending=False)
  selected_features = feature_scores['Feature'][:k].tolist()
  df_x = df_x[selected_features]
  df_x_test = df_x_test[selected_features]
  return df_x, df_x_test


# normalize
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
  scaler = StandardScaler()  # Create a StandardScaler object
  df_normalized = scaler.fit_transform(df)  # Normalize the data
  df_normalized = pd.DataFrame(df_normalized, columns=df.columns)
  return df_normalized


def run_lazy_regressor(df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y['y'], test_size=0.2, random_state=42)
    chosen_regressors = [
        "AdaBoostRegressor",
        "BaggingRegressor",
        "BayesianRidge",
        "DecisionTreeRegressor",
        "DummyRegressor",
        "ElasticNet",
        "ElasticNetCV",
        "ExtraTreeRegressor",
        "ExtraTreesRegressor",
        # 'GammaRegressor', # Did not work
        "GaussianProcessRegressor",
        "GradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "HuberRegressor",
        "KNeighborsRegressor",
        "KernelRidge",
        "Lars",
        "LarsCV",
        "Lasso",
        "LassoCV",
        "LassoLars",
        "LassoLarsCV",
        "LassoLarsIC",
        "LinearRegression",
        "LinearSVR",
        "MLPRegressor",
        "NuSVR",
        "OrthogonalMatchingPursuit",
        "OrthogonalMatchingPursuitCV",
        "PassiveAggressiveRegressor",
        "PoissonRegressor",
        # 'QuantileRegressor', # Did not work
        "RANSACRegressor",
        "RandomForestRegressor",
        "Ridge",
        "RidgeCV",
        "SGDRegressor",
        "SVR",
        "TransformedTargetRegressor",
        "TweedieRegressor",
    ]

    regressors = [
        est[1] for est in all_estimators() if (issubclass(est[1], RegressorMixin) and (est[0] in chosen_regressors))
    ]
    regressors.append(XGBRegressor)
    # regressors.append(lgbm.LGBMRegressor)

    regressor = LazyRegressor(
        verbose=1,
        ignore_warnings=False,
        custom_metric=None,
        predictions=True,
        regressors=regressors,
    )
    print(type(y_train))
    models, _ = regressor.fit(x_train, x_test, y_train, y_test)
    log.info(models)


def train_regressor_on_all_data(df_x: pd.DataFrame, df_y: pd.DataFrame, df_x_test: pd.DataFrame, df_y_test:pd.DataFrame) -> None:
    # Train on all and predict on X_test
    regressor = ExtraTreesRegressor(
      random_state=42,
      n_estimators=200,
    )
    regressor.fit(df_x, df_y['y'])
    df_y_test['y'] = regressor.predict(df_x_test)
    print(df_y_test.head())
    df_y_test.to_csv('X_test_predicted.csv')


def train_regressor_with_tuning(df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y['y'], test_size=0.2, random_state=42)

    regressor = ExtraTreesRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200, 300], # 50, 100, 200
        'max_depth': [None], # , 10, 20
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)

    best_regressor = grid_search.best_estimator_
    y_pred = best_regressor.predict(x_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print("R-squared (R2) score:", r2)



if __name__ == "__main__":
  df_x, df_y, df_x_test = load_data()

  df_x = impute_missing_values(df_x)
  df_x, df_y = remove_outliers(df_x, df_y)
  df_x_normalized = normalize_df(df_x)

  df_x_test = impute_missing_values(df_x_test)
  df_x_test_normalized = normalize_df(df_x_test)

  # feature_selection_with_elbow_method(df_x=df_x_normalized, df_y=df_y, df_x_test=df_x_test_normalized)
  df_x_normalized, df_x_test_normalized = feature_selection(df_x_normalized, df_y, df_x_test_normalized)

  # run_lazy_regressor(df_x=df_x_normalized, df_y=df_y) # best: ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
  df_y_test = pd.DataFrame({'id': df_x_test['id']})
  # train_regressor_with_tuning(df_x_normalized, df_y)
  train_regressor_on_all_data(df_x_normalized, df_y, df_x_test_normalized, df_y_test)

