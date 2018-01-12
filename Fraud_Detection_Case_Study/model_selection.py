import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from clean_data import load_data

def plot_cross_validation(X, y, models, scoring):
    """
    Return 5-Fold Cross Validation scores for various models in addition to
    box plots for each of the 5 fold models.
    Args:
        X: Feature matrix
        y: Target vector
        models: Dictionary of models with the model name as the key and the
        instantiated model as the value.
        scoring: Str of the scoring to use (i.e., 'accuracy')
    Returns:
        Scores: 5-Fold Cross Validation scores for all models.
        Plot: Boxplot of all 5-fold model scores.
    """
    seed = 123
    results = []
    names = []
    all_scores = []
    print('Mod - Avg - Std Dev')
    print('---   ---   -------')
    for name, model in models.items():
        kfold = StratifiedKFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        all_scores.append(cv_results.mean())
        print('{}: {:.2f} ({:2f})'.format(name, cv_results.mean(), cv_results.std()))
    print('Avg of all: {:.3f}'.format(np.mean(all_scores)))
    results = np.array(results)
    df = pd.DataFrame(results.T, columns = names)
    series = df.stack()
    df = series.to_frame(name = 'results').reset_index()
    fig, ax = plt.subplots(figsize = (10, 7))
    sns.boxplot(x="level_1", y="results", data=df, order=['XGBClassifier','Linear SVC','Gradient Boosting',
                'Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest', 'AdaBoost'])
    sns.set_style("darkgrid")
    sns.set(font_scale = 2)
    ax.set_ylabel('{0}'.format(scoring.title()))
    ax.set_xlabel('')
    ax.set_xticklabels(names, rotation=20, fontsize=12)
    ax.set_ylim([.5,1.05])
    ax.set_title('Model Comparision of Stratified 5-Fold CV Scores')
    plt.tight_layout()
    plt.show()

def grid_search(model, X_train, y_train, params, scoring, cv):
    """
    Gridsearch on a model to discover optimized parameters

    Args:
        model: fitted model
        X_train: training feature matrix
        y_train: training target vector
        params: dictionary of model arguments and various values to gridsearch on
        scoring: scoring metric to evaluate the model with
        cv: number of folds in cross validation

    Returns:
        grid_cv.best_score_: The score of the best-performing model
        grid_cv.best_params_: The parameters of the best-performing model
    """
    print('running grid search...')
    grid_cv = GridSearchCV(model, params, n_jobs=-1, scoring=scoring, cv=cv)
    print('fitting grid search...')
    grid_cv.fit(X_train, y_train)
    print('Done.')
    return grid_cv.best_score_, grid_cv.best_params_

def plot_roc(X,y):
    """
    Create Plot of ROC Curve for final Random Forest model

    Args:
        X: training data feature matrix
        y: training data target vector

    Returns:
        ROC Curve
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=fraud)
    rf = RandomForestClassifier(bootstrap=True, criterion='entropy',
        max_depth=None, max_features=10, min_samples_leaf=1,
        min_samples_split=2, n_estimators=25)
    rf.fit(X_train, y_train)
    y_score = rf.predict_proba(X_test)
    y_score = y_score[:,1]
    print('predict probabilities done...')
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    print('roc_curve done...')
    roc_auc = auc(fpr,tpr)
    print('auc done...')
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest Fraud Prediction ROC Curve')
    plt.show()

if __name__ == '__main__':
    # Load Data
    cleandf, fraud = load_data()
    # Perform SMOTE-Overbalancing
    sm = SMOTE(random_state=42)
    X, y = sm.fit_sample(cleandf,fraud)
    #Cross Validate on Various Classification Model
    models = {'XGBClassifier': XGBClassifier(), 'Linear SVC': LinearSVC(),
    'Gradient Boosting': GradientBoostingClassifier(),'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=9),'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),'Random Forest': RandomForestClassifier(), 'AdaBoost': AdaBoostClassifier()}
    possible_score_metrics = ['accuracy','f1','precision','recall','roc_auc']
    plot_cross_validation(X, y, models, 'accuracy') #Random Forest 96%
    plot_cross_validation(X, y, models, 'f1') #Random Forest 95%
    plot_cross_validation(X, y, models, 'precision') #Random Forest 99%
    plot_cross_validation(X, y, models, 'recall') #Random Forest 93%
    plot_cross_validation(X, y, models, 'roc_auc') #Random Forest 98%
    # Gridsearch on Final Model
    model = RandomForestClassifier()
    params = {"n_estimators": [5,10,25,50],"criterion": ("gini", "entropy"),
              "max_features": [1, 3, 10],"max_depth": [3, None],
              "min_samples_split": [2, 3, 10],"min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}
    grid_search(model, X, y, params, 'recall', 3)
    # ROC Curve of Final Model
    plot_roc(X,y)
