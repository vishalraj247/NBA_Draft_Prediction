from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import stats

class Feature:

    def __init__(self):
        # Initialize the random forest classifier with a fixed random state for reproducibility
        self.clf = RandomForestClassifier(random_state=42)
        # List to store selected features based on importance and correlation
        self.selected_features = []

    def feature_engineering(self, df):

        # Temporal Transformation
        df.loc[:, "year_sin"] = np.sin(2 * np.pi * (df["year"] - 1951) / 72)
        df.loc[:, "year_cos"] = np.cos(2 * np.pi * (df["year"] - 1951) / 72)

        
        # Polynomial and Interaction Features
        df['GP_Min_per'] = df['GP'] * df['Min_per']
        df['Ortg_eFG'] = df['Ortg'] * df['eFG']
        
        return df
    
    def handle_outliers(self, df):
        # Store the original number of rows
        original_shape = df.shape[0]
        
        # Compute the z-scores for numerical columns
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        
        # Create a boolean mask to identify rows with z-score less than 5 for all columns
        mask = (z_scores < 5).all(axis=1)
        
        # Apply the mask to remove outliers
        df = df[mask]
        
        # Store the indices of rows that were kept
        indices_kept = df.index
        
        # Compute the new number of rows
        new_shape = df.shape[0]
        
        # Print the number of rows removed
        print(f"Removed {original_shape - new_shape} rows based on z-score.")
        
        return df, indices_kept

    def feature_selection(self, X, y, threshold=0.007, correlation_threshold=0.95):
        # Fit the classifier to the data
        self.clf.fit(X, y)
        
        # Extract feature importances from the trained classifier
        feature_importances = self.clf.feature_importances_
        
        # Identify important features based on the threshold
        important_features = X.columns[feature_importances > threshold].tolist()
        
        # Initialize a set to keep track of correlated features to be removed
        correlated_features = set()
        
        # Compute the correlation matrix only for important features
        correlation_matrix = X[important_features].corr()
        
        # Iterate through the correlation matrix to identify correlated features
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        
        # Remove correlated features from the list of important features
        self.selected_features = [feature for feature in important_features if feature not in correlated_features]
        
        return self.selected_features