import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cleaning import main_clean
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
plt.style.use('fivethirtyeight')

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
    ax.set_ylim([0.5,1])
    ax.set_ylabel('10-Fold CV F1 Score')
    ax.set_xlabel('Model')
    plt.tight_layout()
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

def GradientBoosting_ROC(X_train,X_test,y_train, y_test):
    """
    Plot of ROC Curve for final Gradient Boosting Model.

    Args:
        X_train: training data containing all features
        X_test: testing data containing all features
        y_train: training data containing target
        y_test: testing data containing target

    Returns:
        ROC Plot
    """
    gb = GradientBoostingClassifier(max_depth=4,
                      max_features=None,
                      learning_rate=.25,
                      min_samples_leaf=100,
                      n_estimators=90)
    gb.fit(X_train,y_train)
    y_score = gb.predict_proba(X_test)
    y_score = y_score[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    threshold = thresholds[2400]
    tpr_pt = tpr[2400]
    fpr_pt = fpr[2400]
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('FPR: {:.2f}, TPR: {:.2f}, Threshold: {:.2f}'.format(fpr_pt,
                    tpr_pt,threshold),xy=(fpr_pt,tpr_pt),xytext=(0.4, .8),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gradient Boosting Churn Prediction ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # Load Data
    main_clean(train=True)
    main_clean(train=False)
    train = pd.read_pickle('data/train_data.pkl')
    test = pd.read_pickle('data/test_data.pkl')

    # Test/Train Split with over-sampling using SMOTE
    y = train.pop('Active').values
    X = train.values
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                        test_size=0.33, stratify=y_res)

    # Cross Validate Vanilla Models
    models = {'Logistic Regression': LogisticRegression(), 'K-Nearest Neighbors':
    KNeighborsClassifier(),'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(), 'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(), 'Gradient Boosting': GradientBoostingClassifier(),
    'SVC': SVC()}
    plot_cross_validation(X_train, y_train, models, 'f1')

    # Gridsearch Random Forest
    param_list = {"max_depth": [None],
                  "max_features": [0.25, 0.5, 0.75],
                  "min_samples_split": [2, 10, 20],
                  "min_samples_leaf": [100, 200],
                  "bootstrap": [True, False],
                  "n_estimators" :[100, 200],
                  "criterion": ['gini', 'entropy']}
    test_params(RandomForestClassifier(), X_train, y_train, param_list, 'f1')
    # Best Params: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None,
    # 'max_features': 0.5, 'min_samples_leaf': 100, 'min_samples_split': 2,
    # 'n_estimators': 200}

    # Gridsearch AdaBoost
    param_list = {'n_estimators': [10, 25, 50, 75],
                    'learning_rate':[1.0, 0.5, 0.1]}
    test_params(AdaBoostClassifier(), X_train, y_train, param_list, 'f1')
    # Best Params: {'learning_rate': 1.0, 'n_estimators': 75}

    # Gridsearch Gradient Boosting
    param_list = {'max_depth': [3,4],
                      'max_features': ['sqrt', None],
                      'learning_rate': [ .2,.25, .3],
                      'min_samples_leaf': [100],
                      'n_estimators': [ 70,90],
                      'random_state': [1]}
    test_params(GradientBoostingClassifier(), X_train, y_train, param_list, 'f1')
    # Best Params: {'learning_rate': 0.25, 'max_features': None,
    # 'min_samples_leaf': 100, 'n_estimators': 90}

    # ROC of Gradient Boosted Model on Test/Train Split Training Data
    GradientBoosting_ROC(X_train,X_test,y_train, y_test)

    # Gradient Boosted Model on test set of training data
    gb = GradientBoostingClassifier(max_depth=4,
                      max_features=None,
                      learning_rate=.25,
                      min_samples_leaf=100,
                      n_estimators=90)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print(f1_score(y_test, y_pred))

    # Final Model w/ Test Data
    y_test = test.pop('Active')
    X_test = test
    gb = GradientBoostingClassifier(max_depth=4,
                      max_features=None,
                      learning_rate=.25,
                      min_samples_leaf=100,
                      n_estimators=90)
    gb.fit(X_res, y_res)
    y_pred = gb.predict(X_test)
    print(f1_score(y_test, y_pred))
