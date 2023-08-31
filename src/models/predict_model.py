import pandas as pd
import numpy as np

def predict_with_features(test_data_scaled, original_test_data, player_ids, model, selected_features):
    # Convert numpy array to DataFrame
    test_data_scaled = pd.DataFrame(test_data_scaled, columns=selected_features)
    
    # Use the model to predict probabilities
    test_probs = model.predict_proba(test_data_scaled)[:, 1]
    
    # Create the submission DataFrame
    submission = pd.DataFrame({
        'player_id': player_ids,
        'drafted': test_probs
    })
    
    return submission


def predict_without_features(test_data_scaled, original_test_data, model):
    
    # Use the model to predict probabilities for the test dataset
    test_probs = model.predict_proba(test_data_scaled)[:, 1]
    
    # Create a submission dataframe with player IDs and their corresponding predicted probabilities
    submission = pd.DataFrame({
        'player_id': original_test_data['player_id'].values, 
        'drafted': test_probs
    })
    
    return submission