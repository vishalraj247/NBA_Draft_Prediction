import pandas as pd

def load_data():
    # Load datasets.
    train_data = pd.read_csv('../data/raw/train.csv', low_memory=False)
    test_data = pd.read_csv('../data/raw/test.csv')
    sample_submission = pd.read_csv('../data/raw/sample_submission.csv')
    metadata = pd.read_csv('../data/raw/metadata.csv')
    
    return train_data, test_data, sample_submission, metadata

def display_head(train_data, test_data, sample_submission, metadata):
    # Display the first few rows of each dataset.
    train_head = train_data.head()
    test_head = test_data.head()
    sample_submission_head = sample_submission.head()
    metadata_head = metadata.head()
    
    return train_head, test_head, sample_submission_head, metadata_head

def missing_values_analysis(train_data):
    # Analyze missing values in the train dataset.
    missing_vals = train_data.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    missing_vals_percentage = (missing_vals / len(train_data)) * 100
    
    return missing_vals, missing_vals_percentage

def target_distribution(train_data):
    # Check the distribution of the target variable 'drafted'.
    return train_data['drafted'].value_counts(normalize=True)