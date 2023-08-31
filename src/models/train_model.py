from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self, model_type="RandomForest"):
        # Initialize the model type attribute
        self.model_type = model_type
        
        # Create the classifier based on the specified model type
        if model_type == "RandomForest":
            # Initialize the random forest classifier with a fixed random state for reproducibility
            self.clf = RandomForestClassifier(random_state=42)
        elif model_type == "XGBoost":
            # Initialize the XGBoost classifier with fixed parameters
            self.clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        # Placeholder to store the best model obtained after grid search
        self.best_model = None

    def train(self, X, y):
        # Fit the model to the training data
        self.clf.fit(X, y)

    def grid_search(self, X, y, param_grid):
        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(estimator=self.clf, param_grid=param_grid,
                                   scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
        # Fit the grid search to the data
        grid_search.fit(X, y)
        
        # Store the best model obtained from grid search
        self.best_model = grid_search.best_estimator_
        
        return self.best_model

    def predict_proba(self, X):
        # Use the best model to predict probabilities for the positive class
        return self.best_model.predict_proba(X)[:, 1]