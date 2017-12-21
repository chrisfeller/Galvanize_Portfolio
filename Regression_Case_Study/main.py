import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet

class Pipeline():

    def __init__(self):
        pass

    def load_data(self):
        """
        Loads the training and testing data from either a .csv file or .pkl
        file, depending on if the data has been cleaned and pickled previously.

        Args:
            None

        Returns:
            train: Training dataset
            test: testing dataset
        """
        if os.path.exists('data/train.pkl') and os.path.exists('data/test.pkl'):
            train = pd.read_pickle('data/train.pkl')
            test = pd.read_pickle('data/test.pkl')
        else:
            train = pd.read_csv('data/train.csv')
            test = pd.read_csv('data/test.csv')
        return train, test

    def clean_data(self, df):
        """
        Cleans data by dropping columns made up of 25% or more null values, in
        addition to insignificant or duplicate features. Converts features to
        their correct datatypes. Replaces one specific outlier in the YearMade
        feature. Fills nulls in the Enclosure feature. Creates an Age feature
        based on the saledate. Dummies all categorical variables.

        Args:
            df: Uncleaned dataframe

        Returns:
            df: Cleaned dataframe
        """
        # Drop columns w. >= 25% null values
        df = df[df.columns[df.isnull().mean() < .25]]
        # Convert incorrect dtype
        df['saledate'] = pd.to_datetime(df['saledate'])

        # Drop insignificant or duplicate columns
        df.drop(['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID',
                'fiModelDesc', 'fiBaseModel', 'fiProductClassDesc', 'ProductGroup',
                'Hydraulics'], axis=1, inplace=True)

        # Replace outlier (1000) in YearMade feature w/ median YearMade
        df['YearMade'].replace(1000, df['YearMade'].median(), inplace=True)

        # Fill null values in Enclosure with 'Not_Specified'
        # Combine multiple 'EROPS AC' categories into one 'EROPS_AC'
        # Replace 'None or Unspecified' category with 'Not_Specified'
        df['Enclosure'].fillna('Not_Specified', inplace=True)
        df['Enclosure'].replace(['EROPS w AC', 'EROPS AC'], 'EROPS_AC', inplace=True)
        df['Enclosure'].replace('None or Unspecified', 'Not_Specified', inplace=True)

        # Create Age Columns
        df['Age'] = df['saledate'].dt.year - df['YearMade']
        df.drop('saledate', axis=1, inplace=True)

        # Turn categorical variables into dummy variables
        df = pd.get_dummies(df, dummy_na=False)
        return df

    def scale_data(self, train, test):
        """
        Scales all continue features.

        Args:
            train: unscaled training data
            test: unscaled testing data

        Returns:
            train: scaled training data
            test: scaled testing data
        """
        scaler = StandardScaler()
        continuous_features = [col for col in train.select_dtypes(include='int64')]
        train[continuous_features] = scaler.fit_transform(train[continuous_features])
        test[continuous_features] = scaler.transform(test[continuous_features])
        return train, test

    def model_selection(self, train):
        """
        Performs a test/train split on the training data. Gridsearches over three
        regularixation models (Lasso, Ridge, and ElasticNet), and fits a final
        model using the best performing model (Ridge) from the gridsearch stage.
        Returns the validation MSE and RMSE of the final model.

        Args:
            train: cleaned and scaled training data

        Returns:
            Validation MSE and RMSE of best performing gridsearched model.
        """
        # Test/Train split training data
        y = train['SalePrice']
        X = train.drop('SalePrice', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Gridsearch Lasso Model
        lasso = Lasso()
        param_list = {'alpha': np.linspace(.1, 1, 10)}
        lasso_grid = GridSearchCV(lasso, param_list, scoring='neg_mean_squared_error',
                         cv=5)
        lasso_grid.fit(X_train, y_train)
        print('Model: {}, Best Params: {}, Best Score: {}'\
            .format(lasso, lasso_grid.best_params_, abs(lasso_grid.best_score_)))

        # Gridsearch Ridge Model
        ridge = Ridge()
        param_list = {'alpha': np.linspace(.1, 1, 10),
                        'solver': ['auto', 'svd', 'lsqr', 'cholesky']}
        ridge_grid = GridSearchCV(ridge, param_list, scoring='neg_mean_squared_error',
                         cv=5)
        ridge_grid.fit(X_train, y_train)
        print('Model: {}, Best Params: {}, Best Score: {}'\
            .format(ridge, ridge_grid.best_params_, abs(ridge_grid.best_score_)))

        # Gridsearch ElasticNet Model
        elastic = ElasticNet()
        param_list = {'alpha': np.linspace(0.5, 0.9, 20),
                      'l1_ratio': np.linspace(0.9, 1.0, 10)}
        elastic_grid = GridSearchCV(elastic, param_list, scoring='neg_mean_squared_error',
                         cv=5)
        elastic_grid.fit(X_train, y_train)
        print('Model: {}, Best Params: {}, Best Score: {}'\
            .format(elastic, elastic_grid.best_params_, abs(elastic_grid.best_score_)))

        # Best model on validation set of training data
        final_ridge = Ridge(alpha=1.0, solver='svd')
        final_ridge.fit(X_train, y_train)
        y_pred = final_ridge.predict(X_test)
        log_diff = np.log(y_pred + 1) - np.log(y_test + 1)
        score = np.sqrt(np.mean(log_diff**2))
        print('Validation MSE Score: {}'.format(mean_squared_error(y_test, y_pred)))
        print('Validation RMSLE Score: {}'.format(score))

    def final_model(self, train, test):
        """
        Prepares the training and testing data and then fits the final model,
        using the same optimized parameters from the previous model_selection
        stage. Returns the test MSE and RMSE of the final model, in addition to
        the top 15 coefficients.

        Args:
            train: cleaned and scaled training data
            test: cleaned and scaled testing data

        Returns:
            Test MSE and RMSE of final model, in addition to the top 15
            coefficients.
        """
        # Prepare Data
        y_train = train['SalePrice']
        X_train = train.drop('SalePrice', axis=1)
        y_test = pd.read_csv('data/ValidSolution.csv', index_col=0)
        X_test = test.drop('SalePrice', axis=1)

        # Fit Model
        final_model = Ridge(alpha=1.0, solver='svd')
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # Print MSE and RMSE
        log_diff = np.log(y_pred + 1) - np.log(y_test['SalePrice'] + 1)
        score = np.sqrt(np.mean(log_diff**2))
        print('Test (Holdout) MSE Score: {}'.format(mean_squared_error(y_test, y_pred)))
        print('Test (Holdout) RMSLE Score: {}'.format(score))

        # Print Top 15 Coefficients
        coefs = list(final_model.coef_)
        features = list(X_train.columns)
        importances = [[x, y] for x, y in zip(features, coefs)]
        importances.sort(key=lambda row: abs(row[1]), reverse=True)
        print('\nTop 15 Coefficients:')
        for pair in importances[:15]:
            print(pair)


if __name__=='__main__':
    # Instantiate Class
    pipe = Pipeline()

    # Load Data
    train, test = pipe.load_data()

    # Clean Data
    cutoff = len(train)
    # Temporarily combine training and test data to more easily clean both
    combined = pd.concat(objs=[train, test], axis=0)
    combined = pipe.clean_data(combined)
    # Split training and test data
    train = combined[:cutoff]
    test = combined[cutoff:]
    # Pickle training and test dataframes
    train.to_pickle('data/train.pkl')
    test.to_pickle('data/test.pkl')

    # Scale Data
    train, test = pipe.scale_data(train, test)

    # Model Selection
    pipe.model_selection(train)
    # Lasso: Best Params: {'alpha': 0.5}, Best Score: 295510459.1264006
    # Ridge: Best Params: {'alpha': 1.0, 'solver': 'svd'},
            # Best Score: 294933308.12204915
    # Elastic: Best Params: {'alpha': 0.52105263157894732, 'l1_ratio': 1.0},
            # Best Score: 296362713.3290253

    # Final Model
    pipe.final_model(train, test)
