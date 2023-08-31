import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import pairwise_distances_argmin_min

class DataPreprocessor:
    
    def __init__(self):
        # Initialize the StandardScaler
        self.scaler = StandardScaler()

    def preprocess_train(self, df):

        # Drop the 'pick' column if it exists
        df.drop(columns=['pick'], inplace=True, errors='ignore')

        # Identify columns with missing values
        missing_values = df.isnull().sum()
        missing_values_cols = [col for col in missing_values.index if missing_values[col] > 0]

        # Segregate columns based on their data type (numeric and non-numeric)
        numeric_cols = df[missing_values_cols].select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = df[missing_values_cols].select_dtypes(exclude=['float64', 'int64']).columns

        # Fill missing values in numeric columns with their median value
        for column in numeric_cols:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)

        # Fill missing values in non-numeric columns with their mode value
        for column in non_numeric_cols:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

        return df

    def preprocess_test(self, df):

        # The logic for preprocessing test data is the same as train data currently
        # This is kept separate for potential modifications specific to test data in the future
        return self.preprocess_train(df)

    def encode(self, df, categorical_cols):
        # One-hot encode the categorical columns for the test data
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df_encoded
    
    def scale(self, df):
        # Scale
        scaled_array = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns)

    def encode_and_scale(self, df, categorical_cols):
        # Encoding and scaling
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df_encoded_scaled = self.scaler.fit_transform(df_encoded.drop(columns=['player_id'], errors='ignore'))
        return df_encoded, df_encoded_scaled
    
    def apply_smote(self, X_encoded, X_encoded_scaled, y):
            
            # Temporarily separate 'player_id' from the dataset
            player_ids = X_encoded['player_id']
            
            # Apply SMOTE on the scaled dataset without 'player_id'
            smote = SMOTE(random_state=42)
            X_resampled_array, y_resampled = smote.fit_resample(X_encoded_scaled, y)
            
            # Convert the resampled data back to DataFrame and get original column names
            X_resampled_df = pd.DataFrame(X_resampled_array, columns=X_encoded.drop(columns=['player_id'], errors='ignore').columns)
            
            # Assign 'player_id' to resampled data based on closest original data points
            closest_indices = pairwise_distances_argmin_min(X_resampled_array, X_encoded_scaled)[0]
            X_resampled_df['player_id'] = player_ids.iloc[closest_indices].values
            
            return X_resampled_df, y_resampled
    
    def apply_smote_scale(self, X_scaled, subset_player_ids, y):

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled_array, y_resampled = smote.fit_resample(X_scaled, y)

            # Convert the resampled data back to DataFrame
            X_resampled_df = pd.DataFrame(X_resampled_array, columns=X_scaled.columns)

            # Attach 'player_id' to resampled data based on closest original data points
            closest_indices, _ = pairwise_distances_argmin_min(X_resampled_array, X_scaled)
            X_resampled_df['player_id'] = subset_player_ids.iloc[closest_indices].values

            return X_resampled_df, y_resampled

    def encode_and_scale_test(self, df, categorical_cols, train_cols):
            
            # One-hot encode the categorical columns for the test data
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

            # Ensure the test data has the same columns as the train data
            missing_cols = set(train_cols) - set(df_encoded.columns) - {'player_id'}
            if missing_cols:
                missing_df = pd.DataFrame(0, index=df_encoded.index, columns=list(missing_cols))
                df_encoded = pd.concat([df_encoded, missing_df], axis=1)

            # Reorder columns to match train data
            df_encoded = df_encoded[train_cols]

            # Scale the dataframe excluding 'player_id' column using the scaler fitted on train data
            df_encoded_scaled = self.scaler.transform(df_encoded.drop(columns=['player_id'], errors='ignore'))

            return df_encoded, df_encoded_scaled

    def encode_and_scale_test_features(self, df, categorical_cols, train_cols, selected_features, features_instance):
        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure the test data has the same columns as the training data
        missing_cols = set(train_cols) - set(df_encoded.columns) - {'player_id'}
        for col in missing_cols:
            df_encoded[col] = 0
        
        # Reorder the columns to match the training data
        df_encoded = df_encoded[train_cols]
        
        # Feature Engineering using external 'features' class
        df_encoded = features_instance.feature_engineering(df_encoded.drop(columns=['player_id'], errors='ignore'))
        
        # Add 'team_Kentucky' if it's not in df_encoded but in selected_features
        if 'team_Kentucky' in selected_features and 'team_Kentucky' not in df_encoded.columns:
            df_encoded['team_Kentucky'] = 0
        
        # Select Features
        df_encoded_selected = df_encoded[selected_features]
        
        # Scale
        df_encoded_scaled = self.scaler.transform(df_encoded_selected.drop(columns=['player_id'], errors='ignore'))
        
        return df_encoded, df_encoded_scaled