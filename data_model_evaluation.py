from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from yellowbrick.model_selection import LearningCurve
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union, List
import joblib
from typing import Optional

def separate_train_test(df, y: str):
    """
    Divides data into training and test sets
    Args:
        df: dataframe with data to be divided
        y: name of a dependent variable

    Returns:
        X_train, X_test, y_train, y_test: four df divided into x and y for training and test sets
    """
    X = df.drop(columns=[y])
    y = df[[y]]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)  # zwraca tupla z czterema elementami
    return X_train, X_test, y_train, y_test

def scaling_reg_train(X, y):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    numeric, category = 0, 0
    try:
        scaled_x = scaler_x.fit_transform(X.select_dtypes(exclude=['object', 'bool']))
        numeric += 1
    except:
        pass
    try:
        encoded_x = ohe.fit_transform(X.select_dtypes(include=['object', 'bool']))
        category += 1
    except:
        pass
    scaled_y = scaler_y.fit_transform(y)
    if numeric == 1 and category == 1:
        processed_data = np.concatenate([scaled_x, encoded_x], axis=1)
        return processed_data, scaled_y, scaler_x, scaler_y, ohe
    elif numeric == 1 and category == 0:
        return scaled_x, scaled_y, scaler_x, scaler_y, ohe
    else:
        return encoded_x, scaled_y, scaler_x, scaler_y, ohe

def quick_rfecv_eval(scaled_x: pd.Series, scaled_y: pd.Series, scaler_x: Optional[StandardScaler] = None,
                      ohe: Optional[OneHotEncoder] = None) -> None:
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) using several regression models to evaluate the
    optimal number of features and their ranking.

    Args:
        scaled_x: A pandas series with the scaled independent variables.
        scaled_y: A pandas series with the scaled dependent variable.
        scaler_x: A standard scaler object fit on the independent variables. Default is None.
        ohe: A one-hot encoder object fit on the independent variables. Default is None.

    Returns:
        None.

    Raises:
        None.
    """
    # Initialize regression models
    lr = LinearRegression()
    lasso = Lasso(alpha=0.1)
    ridge = Ridge()
    svr = SVR(kernel='linear')
    dt = DecisionTreeRegressor(random_state=1)
    rf = RandomForestRegressor(random_state=1)

    # Iterate over regression models
    for name, model in [('Linear Regression', lr), ('Lasso Regression', lasso),
                        ('Ridge Regression', ridge), ('SVR', svr), ('Decision Tree Regression', dt),
                        ('Random Forest Regression', rf)]:
        # Apply RFECV
        rfecv = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error')
        rfecv.fit(scaled_x, scaled_y)

        # Print results
        print(f'Model: {name}')
        print(f'Optimal number of features: {rfecv.n_features_}')
        print(f'Feature ranking: {rfecv.ranking_}')
        if scaler_x and ohe:
            # Print feature names if scaler_x and ohe are provided
            print(f'Features names: {[item[1] for item in [(rank, name) for rank, name in zip(rfecv.ranking_, np.concatenate([scaler_x.get_feature_names_out(), ohe.get_feature_names_out()]))] if item[0] == 1]}')




          
def save_model(model: Union[object, List[object]], file_path: str) -> None:
    """
    Saves an ML model or a list of models using the joblib library.

    Args:
        model: An ML model object or a list of ML model objects.
        file_path: A string representing the file path to save the model.

    Returns:
        None.

    Raises:
        None.
    """
    # Save single model
    if not isinstance(model, list):
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    # Save multiple models
    else:
        for idx, m in enumerate(model):
            file = file_path.split(".")[0] + f"_{idx}" + ".joblib"
            joblib.dump(m, file)
            print(f"Model {idx} saved to {file}")

def linear_model_evaluation(model, scaled_x: np.ndarray, scaled_y: np.ndarray, scaler_y, params: dict, cv_folds=10):
  linear_pipeline = Pipeline([
    ('pca', PCA()),
    ('polynomial', PolynomialFeatures()),
    ('actual_model', model)
      ])

  gs = GridSearchCV(linear_pipeline,
                           param_grid=params,
                           cv=cv_folds)

  lr_model = gs.fit(scaled_x, scaled_y)

  y_pred = scaler_y.inverse_transform(lr_model.predict(scaled_x).reshape(-1, 1)).ravel()
  y_train_orig = scaler_y.inverse_transform(scaled_y.reshape(-1, 1)).ravel()

  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))

  ax1.scatter(y_train_orig, y_pred)
  ax1.plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], 'k--', lw=3)
  ax1.set_title('Actual vs. predicted plot')
  ax1.set_xlabel('Actual Values')
  ax1.set_ylabel('Predicted Values')

  df_cv_results_ = pd.DataFrame(lr_model.cv_results_)
  y_test_split = df_cv_results_.iloc[lr_model.best_index_].loc[[f'split{i}_test_score' for i in range(0, lr_model.cv)]].values
  x = [x for x in range(0, lr_model.cv)]

  ax2.plot(x, y_test_split)
  ax2.set_xticks(np.arange(0, lr_model.cv))
  ax2.set_title('Cross-validation Test Scores')
  ax2.set_xlabel('Split')
  ax2.set_ylabel('Test Score')

  visualizer = LearningCurve(
    lr_model.best_estimator_, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10), cv=cv_folds)

  visualizer.fit(scaled_x, scaled_y)
  visualizer.ax = ax3
  visualizer.show()

  plt.show()

  results = {'best parameters': lr_model.best_params_,
             'best score': lr_model.best_score_,
             'MAE (on train set)': mean_absolute_error(y_train_orig, y_pred),
             'MAPE (on train set)': mean_absolute_percentage_error(y_train_orig, y_pred),
             'mean_fit_time': df_cv_results_.iloc[lr_model.best_index_][['mean_fit_time']].values[0],
             'std_fit_time': df_cv_results_.iloc[lr_model.best_index_][['std_fit_time']].values[0],
             'all_best_split_scores': df_cv_results_.iloc[lr_model.best_index_].loc[[f'split{i}_test_score' for i in range(0, lr_model.cv)]].values}

  print('\n')
  for key, value in results.items():
    print(key, ':', value)
  return lr_model, results

def decision_tree_model_evaluation(model, scaled_x: np.ndarray, scaled_y: np.ndarray, scaler_y, params: dict, cv_folds=10, scaler_x=None, ohe=None,):
  
  linear_pipeline = Pipeline([
    ('pca', PCA()),
    ('polynomial', PolynomialFeatures()),
    ('actual_model', model)
      ])

  tree_grid = GridSearchCV(model,
                           param_grid=params,
                           cv=cv_folds)

  tree_model = tree_grid.fit(scaled_x, scaled_y)

  y_pred = scaler_y.inverse_transform(tree_model.predict(scaled_x).reshape(-1, 1)).ravel()
  y_train_orig = scaler_y.inverse_transform(scaled_y.reshape(-1, 1)).ravel()

  df_cv_results_ = pd.DataFrame(tree_model.cv_results_)

  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))

  ax1.scatter(y_train_orig, y_pred)
  ax1.plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], 'k--', lw=3)
  ax1.set_title('Actual vs. predicted plot')
  ax1.set_xlabel('Actual Values')
  ax1.set_ylabel('Predicted Values')

  df_cv_results_ = pd.DataFrame(tree_model.cv_results_)
  y_test_split = df_cv_results_.iloc[tree_model.best_index_].loc[[f'split{i}_test_score' for i in range(0, tree_model.cv)]].values
  x = [x for x in range(0, tree_model.cv)]

  ax2.plot(x, y_test_split)
  ax2.set_xticks(np.arange(0, tree_model.cv))
  ax2.set_title('Cross-validation Test Scores')
  ax2.set_xlabel('Split')
  ax2.set_ylabel('Test Score')

  visualizer = LearningCurve(
    tree_model.best_estimator_, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10), cv=cv_folds)

  visualizer.fit(scaled_x, scaled_y)
  visualizer.ax = ax3
  visualizer.show()

  plt.show()

  if scaler_x and ohe == None:
    x_bar = [scaler_x.get_feature_names_out()]
    y_bar = tree_model.best_estimator_.feature_importances_
    plt.bar(x_bar,
        y_bar)

    plt.xticks(rotation=90)
    plt.show()
  elif scaler_x == None and ohe:
    x_bar = [ohe.get_feature_names_out()]
    y_bar = tree_model.best_estimator_.feature_importances_
    plt.bar(x_bar,
        y_bar)

    plt.xticks(rotation=90)
    plt.show()
  elif scaler_x and ohe:
    x_bar = np.concatenate([scaler_x.get_feature_names_out(), ohe.get_feature_names_out()])
    y_bar = tree_model.best_estimator_.feature_importances_
    plt.bar(x_bar,
        y_bar)

    plt.xticks(rotation=90)
    plt.show()
  else:
    pass

  results = {'best parameters': tree_model.best_params_,
             'best score': tree_model.best_score_,
             'MAE (on train set)': mean_absolute_error(y_train_orig, y_pred),
             'MAPE (on train set)': mean_absolute_percentage_error(y_train_orig, y_pred),
             'mean_fit_time': df_cv_results_.iloc[tree_model.best_index_][['mean_fit_time']].values[0],
             'std_fit_time': df_cv_results_.iloc[tree_model.best_index_][['std_fit_time']].values[0],
             'all_best_split_scores': df_cv_results_.iloc[tree_model.best_index_].loc[[f'split{i}_test_score' for i in range(0, tree_model.cv)]].values}

  print('\n')
  for key, value in results.items():
    print(key, ':', value)
  return tree_model, results

def quick_model_comparision(list_of_results, list_of_names):
  split_scores = [i['all_best_split_scores'] for i in list_of_results]
  times = [i['mean_fit_time'] for i in list_of_results]
  std_dev = [i['std_fit_time'] for i in list_of_results]
  MAE = [i['MAE (on train set)'] for i in list_of_results]
  MAPE = [i['MAPE (on train set)'] for i in list_of_results]

  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 5))
  sns.boxplot(split_scores, ax=ax1)
  ax1.set_xticks(range(len(list_of_names)))
  ax1.set_xticklabels(list_of_names)
  ax1.set_title('Scores by model')
  ax1.set_xlabel('Model')
  ax1.set_ylabel('Test Score')  

  ax2.bar(range(len(list_of_names)), times, align='center', alpha=0.5)
  ax2.errorbar(range(len(list_of_names)), times, yerr=std_dev, fmt='none', capsize=5, color='black')
  for i, v in enumerate(times):
      ax2.text(i, v + 0.1, str(round(v, 3)), color='black', fontweight='bold', ha='center', fontsize=8)
  ax2.set_xticks(range(len(list_of_names)))
  ax2.set_xticklabels(list_of_names)

  ax2.set_title('Fit Time')
  ax2.set_xlabel('Model')
  ax2.set_ylabel('Time')

  ax3.bar(range(len(list_of_names)), MAE, align='center', alpha=0.5)
  for i, v in enumerate(zip(MAE, MAPE)):
    ax3.text(i, v[0] + 10, f'{round(v[0],2)}  {round(v[1]*100,2)}%', color='black', fontweight='bold', ha='center', fontsize=8)
  ax3.set_xticks(range(len(list_of_names)))
  ax3.set_xticklabels(list_of_names)

  ax3.set_title('MAE MAPE')
  ax3.set_xlabel('Model')
  ax3.set_ylabel('MAE')

  plt.show()

  for par, name in zip(list_of_results, list_of_names):
    print(f'{name}: {par["best parameters"]}')