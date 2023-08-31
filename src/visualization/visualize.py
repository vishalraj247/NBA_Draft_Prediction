import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(train_data, features_to_explore=['pts', 'ast', 'treb', 'eFG']):
    #Visualize distributions of selected features.
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Distribution of Selected Features', fontsize=16, y=1.05)

    for i, feature in enumerate(features_to_explore):
        sns.histplot(train_data[feature], ax=axs[i//2, i%2], kde=True)
        axs[i//2, i%2].set_title(f'Distribution of {feature}')
        axs[i//2, i%2].set_xlabel(feature)
        axs[i//2, i%2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, X_encoded, threshold=0.01):
    #Visualize importances of features as determined by a trained model.
    # Get the feature importances from the model
    feature_importances = model.clf.feature_importances_  # Adjusted this line

    # Exclude the 'player_id' column if it's present
    features = X_encoded.columns.values
    if 'player_id' in features:
        features = np.delete(features, np.where(features == 'player_id'))

    # Pair and sort features and their importances
    paired_importances = list(zip(features, feature_importances))
    paired_importances = sorted(paired_importances, key=lambda x: x[1], reverse=True)

    # Filter out features with importance less than the threshold
    filtered_importances = [x for x in paired_importances if x[1] > threshold]

    # Plot
    plt.figure(figsize=(15, 10))
    plt.barh([x[0] for x in filtered_importances], [x[1] for x in filtered_importances])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest')
    plt.gca().invert_yaxis()
    plt.show()