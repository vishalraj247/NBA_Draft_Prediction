import pandas as pd
import numpy as np

def predict_with_features(test_data, model, selected_features):
    """Predict the probabilities of being drafted for the test dataset using selected features.
    
    Args:
    - test_data (DataFrame): The test dataset.
    - model (Model): The trained model object.
    - selected_features (list): List of features to be used for prediction.
    
    Returns:
    - DataFrame: Submission dataframe with player IDs and predicted probabilities.
    """
    
    # Select the relevant features for encoded dataset
    test_encoded_selected = test_data[selected_features]
    
    cols_to_check = test_encoded_selected.columns
    if 'player_id' in cols_to_check:
        cols_to_check = cols_to_check.drop('player_id')

    test_encoded_scaled_selected = test_encoded_selected.values[:, np.isin(cols_to_check, selected_features)]
    
    # Convert the selected and scaled test data to DataFrame for prediction
    test_encoded_selected_df = pd.DataFrame(test_encoded_scaled_selected, columns=selected_features)
    
    # Use the model to predict probabilities for the test dataset
    test_probs_2D = model.predict_proba(test_encoded_selected_df)
    
    # Extract only the probabilities for the positive class
    test_probs = test_probs_2D[:, 1]
    
    # Create a submission dataframe with player IDs and their corresponding predicted probabilities
    submission = pd.DataFrame({'player_id': test_data['player_id'], 'drafted': test_probs})
    
    return submission


def predict_without_features(test_data, model):
    """Predict the probabilities of being drafted for the test dataset without using selected features.
    
    Args:
    - test_data (DataFrame): The test dataset.
    - model (Model): The trained model object.
    
    Returns:
    - DataFrame: Submission dataframe with player IDs and predicted probabilities.
    """
    
    # Ensure 'player_id' is not used in prediction
    test_data_without_id = test_data.drop(columns=['player_id'], errors='ignore')
    
    # Use the model to predict probabilities for the test dataset
    test_probs_2D = model.predict_proba(test_data_without_id)
    
    # Extract only the probabilities for the positive class
    test_probs = test_probs_2D[:, 1]
    
    # Create a submission dataframe with player IDs and their corresponding predicted probabilities
    submission = pd.DataFrame({'player_id': test_data['player_id'], 'drafted': test_probs})
    
    return submission