import pandas as pd
import pickle
from clean_data import load_data
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class MyModel():
    def fit(self, X_train, y_train):
        """
        Perform SMOTE oversampling on training data. Fit final model on
        the resulting training data.

        Args:
            X_train: Training feature matrix
            y_Train: Training target vector

        Returns:
            None
        """
        self.model = RandomForestClassifier(bootstrap=True, criterion='entropy',
            max_depth=None, max_features=10, min_samples_leaf=1,
            min_samples_split=2, n_estimators=25)
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_sample(X_train,y_train)
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_

    def predict(self, X_test):
        """
        Predict fraud on new test data

        Args:
            X_test: Test matrix

        Returns:
            Predictions of fraud for each observation in the test matrix
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predict fraud probabilities on new test data

        Args:
            X_test: Test matrix

        Returns:
            Probabilities of fraud for each observation in the test matrix
        """
        return self.model.predict_proba(X_test)

if __name__ == '__main__':
    # Load Clean Data
    X, y = load_data()
    # Instantiate MyModel Class for Final Model
    model = MyModel()
    # Fit Final model on X_train, y_train
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
