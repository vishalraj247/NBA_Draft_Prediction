from sklearn.ensemble import RandomForestClassifier

class Feature:

    def __init__(self):
        # Initialize the random forest classifier with a fixed random state for reproducibility
        self.clf = RandomForestClassifier(random_state=42)
        # List to store selected features based on importance and correlation
        self.selected_features = []

    def feature_selection(self, X, y, threshold=0.007, correlation_threshold=0.95):
        """Selects important features based on their importance and inter-feature correlation.
        
        Args:
        X (DataFrame): Feature data.
        y (array-like): Target variable.
        threshold (float, optional): Importance threshold to select features. Defaults to 0.01.
        correlation_threshold (float, optional): Correlation threshold to remove correlated features. Defaults to 0.95.
        
        Returns:
        list: List of selected features.
        """
        
        # Fit the classifier and obtain feature importances
        self.clf.fit(X, y)
        feature_importances = self.clf.feature_importances_
        
        # Filter out features based on importance threshold
        important_features = X.columns[feature_importances > threshold].tolist()

        # Identify pairs of features that are highly correlated
        correlated_features = set()
        correlation_matrix = X[important_features].corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        
        # Retain only the important features which are not highly correlated
        self.selected_features = [feature for feature in important_features if feature not in correlated_features]

        return self.selected_features