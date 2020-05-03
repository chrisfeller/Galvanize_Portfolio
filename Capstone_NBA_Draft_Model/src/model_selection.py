import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cleaning import create_train_holdout, clean_train
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def plot_cross_validation(X, y, models, scoring):
    """
    Return 10-Fold Cross Validation scores for various models in addition to
    box plots for each of the 10 fold models.

    Args:
        X: Feature matrix
        y: Target vector
        models: Dictionary of models with the model name as the key and the
        instantiated model as the value.
        scoring: Str of the scoring to use (i.e., 'accuracy')

    Returns:
        Scores: 10-Fold Cross Validation scores for all models.
        Plot: Boxplot of all 10-fold model scores.
    """
    seed = 123
    results = []
    names = []
    all_scores = []
    print('Mod - Avg - Std Dev')
    print('---   ---   -------')
    for name, model in models.items():
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        all_scores.append(cv_results.mean())
        print('{}: {:.2f} ({:2f})'.format(name, cv_results.mean(), cv_results.std()))
    print('Avg of all: {:.3f}'.format(np.mean(all_scores)))
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison of CrossVal Scores')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, rotation=20, fontsize=10)
    # ax.set_ylim([0.5,1])
    ax.set_ylabel('10-Fold CV MSE Score')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def test_params(model, train_X, train_y, param_list, scoring):
    """
    Use grid search to discover optimal parameters for each tested model

    Args:
        model: fitted model
        train_X: training data containing all features
        train_y: training data containing target
        param_list: dictionary of parameters to test and test values
                    (e.g., {'alpha': np.logspace(-1, 1, 50)})

    Returns :
        Best parameter for the model and its score
    """
    g = GridSearchCV(model, param_list, scoring=scoring, cv=10, n_jobs=-1, verbose=1)
    g.fit(train_X, train_y)
    print('Model: {}, Best Params: {}, Best Score: {}'\
        .format(model, g.best_params_, g.best_score_))


if __name__=='__main__':
    # Load Data
    meta_data = pd.read_csv('../data/meta_data.csv')
    college_per100 = pd.read_csv('../data/college_per100.csv')
    college_advance = pd.read_csv('../data/college_advance.csv')
    college_shooting = pd.read_csv('../data/college_shooting.csv')
    nba_advance = pd.read_csv('../data/nba_advance.csv')

    # Create Training Data
    train, _ = create_train_holdout(meta_data, college_per100, college_advance, college_shooting, nba_advance)

    # Clean Train DataFrame
    train = clean_train(train)

    # Create feature matrix X and target vector y
    y = train['NBA_VORP']
    X = train.iloc[:, 8:67]

    # Perform Train/Test Split on Train DataFrame to create Train and Validation Data
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=42)

    # Standardize Train and Validation
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_validation = scaler.transform(X_validation)

    # Create Bootstrapped Data
    X_train = pd.DataFrame(X_train)
    X_train_synthetic, y_train_synthetic = resample(X_train, y_train,
                                                    n_samples=141,
                                                    random_state=42)
    X_train = pd.concat([X_train, X_train_synthetic])
    y_train = pd.concat([y_train, y_train_synthetic])

    # Gridsearch Lasso
    model = Lasso()
    param_list = {'alpha': np.arange(0.1, 5, 0.1), 'tol': np.arange(0.00001,
                  0.1, 0.001), 'warm_start':[True, False]}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'alpha': 0.10000000000000001, 'tol': 0.094009999999999996,
    # 'warm_start': True}, Best Score: -1.2093168722907293

    # Gridsearch Ridge
    model = Ridge()
    param_list = {'alpha': np.arange(0.5, 5, 0.1), 'tol': np.arange(0.00001,
                  0.1, 0.001), 'solver': ['auto', 'svd', 'cholesky', 'lsqr',
                  'sparse_cg', 'sag', 'saga']}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'alpha': 4.5999999999999988, 'solver': 'sag', 'tol':
    # 0.089009999999999992}, Best Score: -1.0478798492975854

    # Gridsearch ElasticNet
    model = ElasticNet()
    param_list = {'alpha': np.arange(0.1, 5, 0.1), 'l1_ratio': np.arange(0.01,
                  0.99, 0.05), 'warm_start': [True, False]}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'alpha': 0.10000000000000001, 'l1_ratio': 0.01,
    # 'warm_start': True}, Best Score: -1.068007619838456

    # Gridsearch K-Nearest Neighbors
    model = KNeighborsRegressor()
    param_list = {'n_neighbors': np.arange(3, 30, 3), 'weights': ['uniform',
                  'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'leaf_size': np.arange(15, 45, 5), 'p': [1, 2]}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'algorithm': 'auto', 'leaf_size': 15, 'n_neighbors': 27,
    # 'p': 1, 'weights': 'distance'}, Best Score: -0.22373257149603262

    # Gridsearch Decision Tree
    model = DecisionTreeRegressor()
    param_list = {'max_depth': np.arange(1, 20, 1), 'min_samples_split':
                  np.arange(2, 10, 1), 'min_samples_leaf': np.arange(2, 10, 1),
                  'max_features':['auto', 'sqrt', 'log2']}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'max_depth': 17, 'max_features': 'sqrt', 'min_samples_leaf':
    #  2, 'min_samples_split': 2}, Best Score: -0.36920454097714733

    # Gridsearch Random Forest
    model = RandomForestRegressor()
    param_list = {'n_estimators': np.arange(10, 1000, 50), 'min_samples_split':
                  np.arange(2, 20, 2), 'max_features':['auto', 'sqrt', 'log2'],
                  'warm_start':[True, False]}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'max_features': 'log2', 'min_samples_split': 2,
    # 'n_estimators': 310, 'warm_start': True}, Best Score: -0.27817375333023436

    # Gridsearch AdaBoostRegressor
    model = AdaBoostRegressor()
    param_list = {'n_estimators': np.arange(10, 1000, 50), 'learning_rate':
                  np.arange(0.1, 1.0, 0.1), 'loss': ['linear', 'square',
                  'exponential']}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'learning_rate': 0.90000000000000002, 'loss': 'exponential',
    # 'n_estimators': 510}, Best Score: -0.4918853427746589

    # Gridsearch Gradient Boosting
    model = GradientBoostingRegressor()
    param_list = {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate':
                  np.arange(0.1, 1.0, 0.1), 'n_estimators': np.arange(10, 1000,
                  50), 'max_depth': np.arange(2, 6, 1), 'warm_start':
                  [True, False]}
    test_params(model, X_train, y_train, param_list, 'neg_mean_squared_error')
    # Best Params: {'learning_rate': 0.30000000000000004, 'loss': 'ls',
    # 'max_depth': 4, 'n_estimators': 410, 'warm_start': False}, Best Score:
    # -0.21602199603662492

    # Cross Validate On Optimized Models
    models = {'Lasso Regularized Regression': Lasso(alpha=0.10000000000000001,
              tol=0.094009999999999996, warm_start=True),'''Ridge Regularized
              Regression''': Ridge(alpha=4.5999999999999988, solver='sag',
              tol=0.089009999999999992), 'Elastic Net Regularized Regression':
              ElasticNet(alpha=0.10000000000000001, l1_ratio=0.01,
              warm_start=True), 'K-Nearest Neighbors':
              KNeighborsRegressor(algorithm='auto', leaf_size=15, n_neighbors=27,
               p=1, weights='distance'), 'Decision Tree':
               DecisionTreeRegressor(max_depth=17, max_features='sqrt',
               min_samples_leaf=2, min_samples_split=2), 'Random Forest':
               RandomForestRegressor(max_features='log2', min_samples_split=2,
               n_estimators=310, warm_start=True), 'AdaBoost':
               AdaBoostRegressor(learning_rate=0.90000000000000002,
               loss='exponential', n_estimators=510), 'Gradient Boosting':
               GradientBoostingRegressor(learning_rate=0.30000000000000004,
               loss='ls', max_depth=4, n_estimators=410, warm_start=False)}

    plot_cross_validation(X_train, y_train, models, 'neg_mean_squared_error')
