from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self):
        # Initialize the random forest classifier
        self.clf = RandomForestClassifier(random_state=42)
        # Placeholder for the best model obtained from grid search
        self.best_model = None

    def train(self, X, y):
        """Trains the random forest classifier on provided data.
        
        Args:
        X (DataFrame): Feature data.
        y (array-like): Target variable.
        """
        self.clf.fit(X, y)

    def grid_search(self, X, y, param_grid):
        """Performs grid search to find the best hyperparameters for the model.
        
        Args:
        X (DataFrame): Feature data.
        y (array-like): Target variable.
        param_grid (dict): Dictionary of hyperparameters for grid search.
        
        Returns:
        RandomForestClassifier: The best model found from grid search.
        """
        
        # Setup and perform the grid search
        grid_search = GridSearchCV(estimator=self.clf, param_grid=param_grid,
                                   scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        
        # Store the best model from the search
        self.best_model = grid_search.best_estimator_
        
        return self.best_model

    def predict_proba(self, X):
        """Predicts the probability for the positive class.
        
        Args:
        X (DataFrame): Feature data.
        
        Returns:
        array-like: Predicted probabilities.
        """
        return self.best_model.predict_proba(X)[:, 1]