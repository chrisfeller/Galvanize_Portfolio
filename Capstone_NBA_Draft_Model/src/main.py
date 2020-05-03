import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cleaning import create_train_holdout, clean_train, clean_holdout
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from plotting import plot_residuals, feature_importance_plot












if __name__ == '__main__':
    # Load Data
    meta_data = pd.read_csv('../data/meta_data.csv')
    college_per100 = pd.read_csv('../data/college_per100.csv')
    college_advance = pd.read_csv('../data/college_advance.csv')
    college_shooting = pd.read_csv('../data/college_shooting.csv')
    nba_advance = pd.read_csv('../data/nba_advance.csv')

    # Create Training Data
    train, holdout = create_train_holdout(meta_data, college_per100, college_advance, college_shooting, nba_advance)

    # Clean Train DataFrame
    train = clean_train(train)

    # Clean Hold-Out DataFrame
    holdout = clean_holdout(holdout)

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
                                                    n_samples=141, random_state=42)
    X_train = pd.concat([X_train, X_train_synthetic])
    y_train = pd.concat([y_train, y_train_synthetic])

    # Final Model
    mod = GradientBoostingRegressor(learning_rate=0.30000000000000004, loss='ls',
                                    max_depth=4, n_estimators=410, warm_start=False)
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_validation)
    print(np.sqrt(mean_squared_error(y_validation, y_pred)))
    # 1.08332014066

    # Residual Plot
    plot_residuals(y_validation, y_pred)

    # Feature Importance
    feature_importance_plot(X.columns, mod.feature_importances_)

    # Predictions for Validation Data
    result = pd.concat([y_validation, train], axis=1, join='inner')
    result = pd.concat([result[['sports_ref_name', 'NBA_VORP']].reset_index(),
                                pd.Series(y_pred)], axis=1)
    result = result.iloc[:, [1, 2, -1]]
    result.rename(columns={0:'predicted_NBA_VORP'}, inplace=True)
    result.sort_values(by='predicted_NBA_VORP', ascending=False, inplace=True)
    print(result)

    # Predictions for 2015-16 and 2016-17 Drafts
    y_holdout = holdout['NBA_VORP']
    X_holdout = holdout.iloc[:, 8:67]
    X_holdout = scaler.transform(X_holdout)
    last_two_drafts = mod.predict(X_holdout)
    last_two_draft_result = pd.concat([y_holdout, holdout], axis=1, join='inner')
    last_two_draft_result = pd.concat([last_two_draft_result[['sports_ref_name',
                            'last_college_season']].reset_index(),
                            pd.Series(last_two_drafts)], axis=1)
    last_two_draft_result = last_two_draft_result.iloc[:, 1:]
    last_two_draft_result.sort_values(by=['last_college_season', 0],
                                      ascending=[False, False], inplace=True)
    print(last_two_draft_result)

    # Predictions for 2017-18 Draft
    draft_2018 = pd.read_csv('../data/2017-18_season.csv')
    draft_2018_input = draft_2018.iloc[:, 8:67]
    draft_2018_input = scaler.transform(draft_2018_input)
    predictions_2018 = mod.predict(draft_2018_input)
    results_2018 = pd.concat([draft_2018['sports_ref_name'],
                              pd.Series(predictions_2018)], axis=1)
    results_2018.sort_values(by=0, ascending=False, inplace=True)
    print(results_2018)
