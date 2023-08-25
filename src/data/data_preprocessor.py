import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    
    def __init__(self):
        # Initialize the StandardScaler
        self.scaler = StandardScaler()

    def preprocess_train(self, df):
        """Preprocess training data: handle missing values and drop certain columns.
        
        Args:
        df (DataFrame): The input data frame containing the training data.
        
        Returns:
        DataFrame: The preprocessed data frame.
        """
        
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
        """Preprocess test data: handle missing values and drop certain columns. Note that this is similar
        to the train preprocessing, but they are kept separate for potential future modifications.
        
        Args:
        df (DataFrame): The input data frame containing the test data.
        
        Returns:
        DataFrame: The preprocessed data frame.
        """

        # The logic for preprocessing test data is the same as train data currently
        # This is kept separate for potential modifications specific to test data in the future
        return self.preprocess_train(df)

    def encode_and_scale(self, df, categorical_cols):
        """Encode categorical columns and scale the data.
        
        Args:
        df (DataFrame): The input data frame.
        categorical_cols (list): List of categorical columns to be encoded.
        
        Returns:
        tuple: Tuple containing the encoded dataframe and the scaled version of it.
        """
        
        # One-hot encode the categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Scale the dataframe excluding 'player_id' column
        df_encoded_scaled = self.scaler.fit_transform(df_encoded.drop(columns=['player_id'], errors='ignore'))

        return df_encoded, df_encoded_scaled

    def encode_and_scale_test(self, df, categorical_cols, train_cols):
        """Encode categorical columns and scale the test data.
        
        Args:
        df (DataFrame): The input data frame containing the test data.
        categorical_cols (list): List of categorical columns to be encoded.
        train_cols (list): List of columns present in training data (for consistency).
        
        Returns:
        tuple: Tuple containing the encoded dataframe and the scaled version of it.
        """
        
        # One-hot encode the categorical columns for the test data
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Ensure the test data has the same columns as the train data
        # Add missing columns from train data to test data and set them to 0
        missing_cols = set(train_cols) - set(df_encoded.columns) - {'player_id'}
        for col in missing_cols:
            df_encoded[col] = 0

        # Ensure the order of columns in test data matches with train data
        df_encoded = df_encoded[train_cols]

        # Scale the dataframe excluding 'player_id' column using the scaler fitted on train data
        df_encoded_scaled = self.scaler.transform(df_encoded.drop(columns=['player_id'], errors='ignore'))

        return df_encoded, df_encoded_scaled